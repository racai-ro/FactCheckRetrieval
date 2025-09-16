import torch
from sentence_transformers import SentenceTransformerModelCardData, SentenceTransformer
from peft import LoraConfig, TaskType, PromptTuningConfig, PromptTuningInit
from transformers import BitsAndBytesConfig, AutoConfig

from src.modules.query_only_prompt_st import QueryOnlyPromptSentenceTransformer
from src.utils.query_only_prompt_collator import QueryOnlyPromptCollator


def setup_model(args, data):
    """Set up the model based on the tuning method (Full, LoRA or Prompt Tuning)."""
    if QueryOnlyPromptSentenceTransformer is None or QueryOnlyPromptCollator is None:
        raise RuntimeError("Required prompt tuning modules could not be imported. Cannot set up model.")
        
    model_card_data = SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="Multilingual Fact Retrieval Gemma",
    )
    model_kwargs = {
        "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.bfloat16,
    }
    collator = None

    # Configure quantization if requested
    if args.quantization != 'none':
        print(f'Using {args.quantization} quantization')
        if args.quantization == '4bit':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:  # 8bit
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["quantization_config"] = quantization_config

    if args.method == "prompt":  # prompt method
        model_config = AutoConfig.from_pretrained(args.model_id)
        model = QueryOnlyPromptSentenceTransformer(
            args.model_id,
            model_kwargs=model_kwargs,
            model_card_data=model_card_data,
            prompt_tuning_config=PromptTuningConfig(
                prompt_tuning_init=PromptTuningInit.TEXT,
                prompt_tuning_init_text=args.prompt,
                tokenizer_name_or_path=args.model_id,
                token_dim=model_config.hidden_size,
            )
        ).to(data['device'])

        # Freeze the model except for prompt embeddings
        for param in model.parameters():
            param.requires_grad = False
        model._prompt_embedding.learned_embedding.requires_grad = True

        # Print trainable parameters
        for name, param in model.named_parameters():
            # Ensure input embeddings are frozen for prompt tuning
            if name == "0.auto_model.embed_tokens.weight":
                param.requires_grad = False
            if param.requires_grad:
                print(f'{name} is trainable')

        train_batch_size = args.batch_size
        eval_batch_size = args.batch_size
        collator = QueryOnlyPromptCollator(tokenize_fn=model.tokenize)
        model.max_seq_length = args.max_seq_length
        return model, train_batch_size, eval_batch_size, collator

    model = SentenceTransformer(
        args.model_id,
        model_kwargs=model_kwargs,
        model_card_data=model_card_data,
    ).to(data['device'])

    if args.method == "lora":
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )
        model.add_adapter(lora_config)

    # Print trainable parameters info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    train_batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    model.max_seq_length = args.max_seq_length
    return model, train_batch_size, eval_batch_size, collator




