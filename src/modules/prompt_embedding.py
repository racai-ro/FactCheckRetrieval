import torch
from transformers import AutoTokenizer
import os


class PromptEmbedding(torch.nn.Module):
    def __init__(self, config=None, word_embeddings=None):
        super().__init__()
        tokenizer_kwargs = config.tokenizer_kwargs or {}
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path, **tokenizer_kwargs)
        init_text = config.prompt_tuning_init_text
        init_token_ids = self.tokenizer(init_text, add_special_tokens=False)["input_ids"]
        self.total_virtual_tokens = len(init_token_ids)
        self.word_embeddings = word_embeddings

        init_token_ids = torch.LongTensor(init_token_ids).to(word_embeddings.weight.device)
        word_embedding_weights = word_embeddings(init_token_ids)
        self.learned_embedding = torch.nn.Parameter(word_embedding_weights, requires_grad=True)

    def _get_special_tokens_and_embeddings(self) -> tuple[dict, dict]:
        """Get special token IDs and their embeddings."""

        tokenizer = self.tokenizer
        device = self.word_embeddings.weight.device

        # These tokens depend on model type
        special_tokens = {
            'start': tokenizer.convert_tokens_to_ids(tokenizer.bos_token),
            'instruct': tokenizer.convert_tokens_to_ids("<instruct>"),
            'query': tokenizer.convert_tokens_to_ids("<query>")
        }
        special_tokens = {k: torch.tensor([v], device=device) for k, v in special_tokens.items()}

        special_embeds = {
            k: self.word_embeddings(v)
            for k, v in special_tokens.items()
        }

        return special_tokens, special_embeds

    def forward(self, tokens):
        input_embedding = self.word_embeddings(tokens)
        _, special_embeds = self._get_special_tokens_and_embeddings()

        batch_size = tokens.size(0)

        # Detach special embeddings to ensure they don't get gradients
        # and expand them to batch size
        start_embeds = special_embeds['start'].detach().expand(batch_size, -1, -1)
        instruct_embeds = special_embeds['instruct'].detach().expand(batch_size, -1, -1)
        query_embeds = special_embeds['query'].detach().expand(batch_size, -1, -1)

        # Add our trainable embeddings (these will have requires_grad=True)
        learned_embeds = self.learned_embedding.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate all embeddings
        learned_embedding = torch.cat([
            start_embeds,
            instruct_embeds,
            learned_embeds,  # This tensor will have requires_grad=True
            query_embeds
        ], 1)

        # Concatenate learned embeddings with the input embeddings
        concat = torch.cat([learned_embedding, input_embedding[:, 1:, :]], 1)

        return concat

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'prompt_embedding_state_dict': self.state_dict(),
        }, os.path.join(path, "prompt_embedding.pth"))
