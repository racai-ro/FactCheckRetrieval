from sentence_transformers import SentenceTransformer
from peft import PromptTuningConfig
from src.modules.prompt_embedding import PromptEmbedding
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from sentence_transformers.util import batch_to_device, truncate_embeddings
from sentence_transformers.quantization import quantize_embeddings
import os
from typing import Literal
import logging
from torch import Tensor

logger = logging.getLogger(__name__)

class QueryOnlyPromptSentenceTransformer(SentenceTransformer):
    """
    A subclass of SentenceTransformer that prepends learned prompt embeddings if `is_query` is True.
    Calls super().forward(...) so that the normal modules execute in sequence.
    """

    def __init__(self, *args, prompt_tuning_config: PromptTuningConfig = None, max_seq_length: int = 512, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_seq_length = max_seq_length
        self.prompt_tuning_config = prompt_tuning_config
        self._prompt_embedding = None
        if prompt_tuning_config is not None:
            self._prompt_embedding = PromptEmbedding(
                prompt_tuning_config,
                self[0].auto_model.get_input_embeddings(),  # word_embeddings
            )

    def _pad_sequences(
            self,
            embeds: torch.Tensor,
            attn: torch.Tensor,
            max_batch_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad sequences in a batch to match the batch's maximum length."""
        _, seq_len, _ = embeds.shape
        if seq_len > max_batch_len:
            return embeds[:, :max_batch_len, :], attn[:, :max_batch_len]

        pad_len = max_batch_len - seq_len
        # Pad on the sequence length dimension (dimension 1)
        padded_embeds = F.pad(embeds, (0, 0, 0, pad_len, 0, 0))
        padded_attn = F.pad(attn, (0, pad_len), value=0)
        
        return padded_embeds, padded_attn

    def _compute_embeddings_and_attention(self, features: dict[str, torch.Tensor], mask) -> tuple[
        torch.Tensor, torch.Tensor]:
        """
        Computes input embeddings and attention masks for the batch, handling queries and regular inputs differently.
        """
        device = next(self.parameters()).device
        input_ids = features["input_ids"].to(device)

        input_ids = input_ids[:, 1:]

        # Split input_ids into two tensors based on the mask
        query_input_ids = input_ids[mask]
        non_query_input_ids = input_ids[~mask]

        # Apply self._prompt_embedding to the query part
        query_embeds = self._prompt_embedding(query_input_ids)
        non_query_embeds = self[0].auto_model.embed_tokens(non_query_input_ids)

        # Get the hidden dimension from the embedding dimension (last dimension)
        hidden_dim = query_embeds.shape[-1]
        query_attn = torch.ones((query_embeds.size(0), query_embeds.size(1)), dtype=torch.long, device=device)
        non_query_attn = torch.ones((non_query_embeds.size(0), non_query_embeds.size(1)), dtype=torch.long, device=device)
        
        # Pad sequences to max_length
        query_embeds, query_attn = self._pad_sequences(
            query_embeds, query_attn, self.max_seq_length)
        non_query_embeds, non_query_attn = self._pad_sequences(
            non_query_embeds, non_query_attn, self.max_seq_length)
        
        # Now both query_attn and non_query_attn should have the same sequence length dimension (512)
        embeds = torch.cat([query_embeds, non_query_embeds], dim=0)
        attn = torch.cat([query_attn, non_query_attn], dim=0)
        
        # Reorder embeds and attn to match the original input_ids order
        embeds = embeds[torch.argsort(torch.cat([torch.where(mask)[0], torch.where(~mask)[0]]))]
        attn = attn[torch.argsort(torch.cat([torch.where(mask)[0], torch.where(~mask)[0]]))]
        return embeds, attn

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        mask = (features["input_ids"][:, 0] == -137).to(device=features["input_ids"].device)
        input_embeds, attention_mask = self._compute_embeddings_and_attention(features, mask=mask)
        dtype = self[0].auto_model.dtype
        
        # Process through model modules
        out_features = None
        for module_name, module in self.named_children():
            if module_name == "_prompt_embedding":
                continue
            if out_features is None:
                out_features = module({"inputs_embeds": input_embeds.to(dtype=dtype), "attention_mask": attention_mask.to(dtype=dtype)})
            else:
                out_features = module(out_features)
    
        return out_features

    def encode(
            self,
            sentences: str | list[str],
            batch_size: int = 32,
            show_progress_bar: bool | None = None,
            output_value: Literal["sentence_embedding", "token_embeddings"] | None = "sentence_embedding",
            precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            device: str = None,
            normalize_embeddings: bool = False,
            **kwargs,
    ) -> list[Tensor] | np.ndarray | Tensor:

        self.eval()
        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (logging.INFO, logging.DEBUG)

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
                sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        extra_features = {}

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            query = torch.tensor(['[QUERY]' in sen for sen in sentences_batch], dtype=torch.bool)
            sentences_batch = [sen.replace('[QUERY]', '') for sen in sentences_batch]

            features = self.tokenize(sentences_batch)

            # Only add -137 token for query inputs
            batch_size = features["input_ids"].shape[0]
            prefix = torch.where(
                query.unsqueeze(1),
                torch.tensor([-137]),
                torch.tensor([[0]])
            )

            features["input_ids"] = torch.cat([prefix, features["input_ids"]], dim=1)
            features = batch_to_device(features, device)
            features.update(extra_features)

            with torch.no_grad():
                out_features = self.forward(features, **kwargs)
                out_features["sentence_embedding"] = truncate_embeddings(
                    out_features["sentence_embedding"], self.truncate_dim
                )

                if output_value == "token_embeddings":
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features["attention_mask"]):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0: last_mask_id + 1])
                elif output_value is None:  # Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features["sentence_embedding"])):
                        row = {name: out_features[name][sent_idx] for name in out_features}
                        embeddings.append(row)
                else:  # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if precision and precision != "float32":
            all_embeddings = quantize_embeddings(all_embeddings, precision=precision)

        if convert_to_tensor:
            if len(all_embeddings):
                if isinstance(all_embeddings, np.ndarray):
                    all_embeddings = torch.from_numpy(all_embeddings)
                else:
                    all_embeddings = torch.stack(all_embeddings)
            else:
                all_embeddings = torch.Tensor()
        elif convert_to_numpy:
            if not isinstance(all_embeddings, np.ndarray):
                if all_embeddings and all_embeddings[0].dtype == torch.bfloat16:
                    all_embeddings = np.asarray([emb.float().numpy() for emb in all_embeddings])
                else:
                    all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        elif isinstance(all_embeddings, np.ndarray):
            all_embeddings = [torch.from_numpy(embedding) for embedding in all_embeddings]

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def load_prompt_embedding_weights(self, path: str):
        """
        Loads saved PromptEmbedding weights into the existing _prompt_embedding instance,
        matching the model's dtype (e.g., bfloat16).
        """
        checkpoint = torch.load(os.path.join(path, "2_PromptEmbedding/prompt_embedding.pth"))
        for key, value in checkpoint["prompt_embedding_state_dict"].items():
            if isinstance(value, torch.Tensor):
                # Cast to the same dtype as the existing learned_embedding (often bfloat16)
                checkpoint["prompt_embedding_state_dict"][key] = value.to(
                    self._prompt_embedding.learned_embedding.dtype
                )
        self._prompt_embedding.load_state_dict(checkpoint["prompt_embedding_state_dict"])

