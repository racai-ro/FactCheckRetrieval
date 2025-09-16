from typing import Any, Dict, Union
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.losses import MatryoshkaLoss, CachedMultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers


class CustomSentenceTransformerTrainer(SentenceTransformerTrainer):
    """
    Custom trainer that logs gradient norms for prompt tuning embeddings
    """
    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """Override training_step to add gradient norm logging"""
        # Regular training step
        loss = super().training_step(model, inputs, num_items_in_batch)

        # Check if model is wrapped in DataParallel or DistributedDataParallel
        model_unwrapped = model.module if hasattr(model, "module") else model

        # Only log gradient norm if using prompt tuning
        if hasattr(model_unwrapped, "_prompt_embedding") and model_unwrapped._prompt_embedding is not None:
            if model_unwrapped._prompt_embedding.learned_embedding.grad is not None:
                grad_norm = model_unwrapped._prompt_embedding.learned_embedding.grad.norm().item()
                self.log({"prompt_grad_norm": grad_norm})

        return loss


def configure_loss(args, model):
    """Configure the training loss function."""
    inner_loss = CachedMultipleNegativesRankingLoss(
        model,
        mini_batch_size=args.mini_batch_size
    )

    if args.method == 'lora' and args.use_matryoshka:
        print('Using Matryoshka loss')
        matryoshka_dimensions = [1024, 768, 512, 256, 128, 64]
        train_loss = MatryoshkaLoss(
            model, loss=inner_loss, matryoshka_dims=matryoshka_dimensions
        )
    else:
        train_loss = inner_loss
    return train_loss


def configure_training_args(args, train_batch_size, eval_batch_size, metric_for_best_model):
    """Configure the SentenceTransformerTrainingArguments."""
    return SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=3,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        max_grad_norm=1.0,
    )


def create_trainer(args, model, training_args, train_dataset, train_loss, evaluator, collator):
    """Create the appropriate trainer."""
    if args.method == 'prompt':
        trainer = CustomSentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            loss=train_loss,
            evaluator=evaluator,
            data_collator=collator,
        )
    else:
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            loss=train_loss,
            evaluator=evaluator,
        )
    return trainer
