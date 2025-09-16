from typing import Any, List, Dict
import torch
from sentence_transformers.data_collator import SentenceTransformerDataCollator


class QueryOnlyPromptCollator(SentenceTransformerDataCollator):
    def __init__(self, tokenize_fn):
        super().__init__(tokenize_fn)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        column_names = list(features[0].keys())

        # We should always be able to return a loss, label or not:
        batch = {}

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        if tuple(column_names) not in self._warned_columns:
            self.maybe_warn_about_column_order(column_names)

        # Extract the label column if it exists
        for label_column in self.valid_label_columns:
            if label_column in column_names:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                column_names.remove(label_column)
                break

        for column_name in column_names:
            # If the prompt length has been set, we should add it to the batch
            if column_name.endswith("_prompt_length") and column_name[: -len("_prompt_length")] in column_names:
                batch[column_name] = torch.tensor([row[column_name] for row in features], dtype=torch.int)
                continue

            rows = [row[column_name].replace('[QUERY]', '') for row in features] if column_name == "anchor" \
                else [row[column_name] for row in features]

            tokenized = self.tokenize_fn(rows)
            for idx, (key, value) in enumerate(tokenized.items()):
                if key == "input_ids":
                    if column_name == "anchor":
                        batch_size = value.shape[0]
                        prefix = torch.full((batch_size, 1), -137, device=value.device)
                        value = torch.cat([prefix, value], dim=1)
                    else:
                        batch_size = value.shape[0]
                        prefix = torch.full((batch_size, 1), 0, device=value.device)
                        value = torch.cat([prefix, value], dim=1)

                batch[f"{column_name}_{key}"] = value

        return batch
