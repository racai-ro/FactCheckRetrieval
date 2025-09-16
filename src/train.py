import random
import torch
from src.utils.data_utils import prepare_data, prepare_queries, create_training_dataset
from src.utils.model_utils import setup_model
from src.utils.training_utils import configure_loss, configure_training_args, create_trainer
from src.utils.eval_utils import create_evaluator
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune Gemma model with different methods')

    # General arguments
    parser.add_argument('--output_dir', type=str, default="multilingual-fact-retrieval-gemma-tuned",
                        help='Output directory for model')
    parser.add_argument('--model_id', type=str, default="BAAI/bge-multilingual-gemma2",
                        help='Base model ID')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--mini_batch_size', type=int, default=16,
                        help='Mini batch size')

    # Method selection
    parser.add_argument('--method', type=str, choices=['lora', 'prompt', 'full'], default='lora',
                        help='Tuning method: lora or prompt')

    # Prompt tuning specific arguments
    parser.add_argument('--query_prefix', type=str,
                        default="Given a social media post as a query, retrieve fact checks that verify or debunk the post.\n",
                        help='Prompt text for tuning (used in prompt tuning method)')

    # LoRA specific arguments
    parser.add_argument('--use_matryoshka', action='store_true',
                        help='Use Matryoshka loss for multi-dimensional embeddings (only for LoRA method)')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help='Evaluation batch size')
    parser.add_argument('--quantization', type=str, choices=['none', '4bit', '8bit'], default='none',
                        help='Quantization method to use with LoRA. Options: none, 4bit, 8bit')
    parser.add_argument('--lora_alpha', type=int, default=128,
                        help='LoRA alpha parameter')
    parser.add_argument('--lora_r', type=int, default=64,
                        help='LoRA r parameter')

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Prepare data
    data = prepare_data(args)
    queries = prepare_queries(args, data)

    # Create train/test query/pair splits
    train_queries = {str(post_id): queries[post_id] for post_id in data['train_keys']}
    test_queries = {str(post_id): queries[post_id] for post_id in data['test_keys']}
    train_pairs = {str(k): set(str(el) for el in data['dataset'].pairs[k]) for k in data['train_keys']}
    test_pairs = {str(k): set(str(el) for el in data['dataset'].pairs[k]) for k in data['test_keys']}

    # Configure evaluator
    evaluator, metric_for_best_model = create_evaluator(
        args, test_queries, data, test_pairs
    )

    # Set up model based on method
    model, train_batch_size, eval_batch_size, collator = setup_model(args, data)

    # Create training data
    train_dataset = create_training_dataset(train_queries, train_pairs, data)

    # Configure loss
    train_loss = configure_loss(args, model)

    # Configure training arguments
    training_args = configure_training_args(
        args, train_batch_size, eval_batch_size, metric_for_best_model
    )

    # Create trainer
    trainer = create_trainer(
        args, model, training_args, train_dataset, train_loss, evaluator, collator
    )

    # Train the model
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # Save the best model
    best_model_path = f"{args.output_dir}/best_model"
    model.save(best_model_path)
    print(f"Best model saved to {best_model_path}")


if __name__ == "__main__":
    main()
