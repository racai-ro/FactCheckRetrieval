from sentence_transformers import SentenceTransformer
from src.dataset import PostsDataset
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftModel
import argparse
import os
from src.modules.query_only_prompt_st import QueryOnlyPromptSentenceTransformer

device = torch.device('cuda')

TRANSLATIONS = False


def prepare_fact_text(fact):
    if TRANSLATIONS:
        return f'{fact["en_title"]} {fact["en_claim"]}'
    return f'{fact["title"]}. {fact["claim"]}'


def prepare_post_text(post, embedding_method):
    text = post['text'] or ''
    en_text = post['en_text'] or ''
    ocr_text = post['ocr_text'] or ''
    ocr_en_text = post['ocr_en_text'] or ''
    if TRANSLATIONS:
        return f'{en_text} {ocr_en_text}'

    if embedding_method == 'prompt':
        return f'[QUERY]{text} {ocr_text}'
    else:
        return f'{text} {ocr_text}'


def create_model(embedding_method, base_model, max_seq_length, adapter_model=None, prompt_model_name=None, prompt_init_text=None, query_prefix=None):
    if embedding_method == 'prompt':
        if not prompt_model_name or not prompt_init_text:
            raise ValueError("prompt_model_name and prompt_init_text must be provided for 'prompt' embedding method.")

        print(f"Using prompt-based embedding method with model: {prompt_model_name}")
        model_config = AutoConfig.from_pretrained(base_model)
        prompt_conf = PromptTuningConfig(
            prompt_tuning_init=PromptTuningInit.TEXT,
            prompt_tuning_init_text=prompt_init_text,
            tokenizer_name_or_path=prompt_model_name,
            token_dim=model_config.hidden_size,
        )
        model = QueryOnlyPromptSentenceTransformer(base_model, prompt_tuning_config=prompt_conf).to(device)
        model.load_prompt_embedding_weights(prompt_model_name)
        if adapter_model:
            print(f"Loading adapter {adapter_model} for prompt method.")
            model.load_adapter(adapter_model)
        else:
            print(f"Loading adapter from {prompt_model_name} for prompt method.")
            model.load_adapter(prompt_model_name)
        model.max_seq_length = max_seq_length

    elif embedding_method == 'standard':
        print(f"Using standard SentenceTransformer method with base model: {base_model}")
        model = SentenceTransformer(base_model, trust_remote_code=True)
        if adapter_model:
            print(f"Loading adapter {adapter_model} for standard method.")
            model.load_adapter(adapter_model)
        model.max_seq_length = max_seq_length
        model.tokenizer.padding_side = "right"
        model = model.to(device)

    else:
        raise ValueError(f"Unknown embedding_method: {embedding_method}")

    return model


def compute_embeddings(items, text_prep_fn, output_file, batch_size, type="facts", **kwargs):
    model = kwargs.get('model')

    embeddings_dict = {}
    texts = []
    ids = []

    for item in tqdm(items, desc=f"Preprocessing texts for {os.path.basename(output_file)}"):
        if type == "facts":
            texts.append(text_prep_fn(item))
        else:
            texts.append(text_prep_fn(item, embedding_method=kwargs['embedding_method']))
        ids.append(item['id'])

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Computing embeddings for {os.path.basename(output_file)}"):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        if kwargs['embedding_method'] == 'prompt':
            batch_embeddings = model.encode(
                batch_texts,
                normalize_embeddings=True,
                convert_to_tensor=True
            )
        else:
            current_prompt = kwargs.get('query_prefix', None)
            batch_embeddings = model.encode(
                batch_texts,
                prompt=current_prompt,
                normalize_embeddings=True,
                convert_to_tensor=True
            )

        for idx, emb in zip(batch_ids, batch_embeddings):
            embeddings_dict[idx] = emb.cpu()

    torch.save(embeddings_dict, output_file)
    print(f"Saved {len(embeddings_dict)} embeddings to {output_file}")
    return embeddings_dict


def get_output_filename(prefix, args):
    method_part = args.embedding_method
    if args.embedding_method == 'prompt':
        model_identifier = args.prompt_model_name.replace("/", "_")
    else:
        model_identifier = args.base_model.replace("/", "_")
        if args.adapter_model and args.adapter_model.lower() != "none":
            model_identifier += f"_adapter_{args.adapter_model.replace('/', '_')}"

    trans_part = f"translations_{args.translations}"
    return f"{prefix}_embeddings_{method_part}_{model_identifier}_{trans_part}.pt"


def main():
    parser = argparse.ArgumentParser(description='Compute embeddings for fact-checking retrieval')
    parser.add_argument('--embedding_method', type=str, default='standard', choices=['standard', 'prompt'],
                        help='Method to use for computing embeddings: standard SentenceTransformer or prompt-based.')
    parser.add_argument('--base_model', type=str, default="BAAI/bge-multilingual-gemma2",
                        help='Base model identifier (used by both methods).')
    parser.add_argument('--adapter_model', type=str, default='multilingual-fact-retrieval/checkpoint-40',
                        help='Adapter model to load for standard method, or optionally for prompt method. Use "none" to skip loading adapter.')
    parser.add_argument('--prompt_model_name', type=str,
                        default='multilingual-fact-retrieval-gemma-prompt/checkpoint-140',
                        help='Model name/path for prompt embeddings and adapter (used only if embedding_method=prompt).')
    parser.add_argument('--prompt_init_text', type=str,
                        default="Given a social media post as a query, retrieve fact checks that verify or debunk the post.\n",
                        help='Initialization text for prompt tuning (used only if embedding_method=prompt).')
    parser.add_argument('--query_prefix', type=str,
                        default="<instruct>Given a social media post as a query, retrieve fact checks that verify or debunk the post.\n<query>",
                        help='Prefix for post queries when using standard sentence transformer encoding.')
    parser.add_argument('--translations', action='store_true',
                        help='Use English translations instead of original text')
    parser.add_argument('--max_seq_length', type=int, default=1024,
                        help='Max sequence length for the model')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for encoding')
    parser.add_argument('--data_folder', type=str, default='data',
                        help='Data folder containing dataset files')

    args = parser.parse_args()

    global TRANSLATIONS
    TRANSLATIONS = args.translations

    adapter = args.adapter_model if args.adapter_model and args.adapter_model.lower() != "none" else None

    dataset = PostsDataset(args.data_folder, phase='dev')

    model = create_model(
        embedding_method=args.embedding_method,
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        adapter_model=adapter,
        prompt_model_name=args.prompt_model_name,
        prompt_init_text=args.prompt_init_text,
        query_prefix=args.query_prefix if args.embedding_method == 'standard' else None
    )

    common_args = {
        'batch_size': args.batch_size,
        'model': model,
        'embedding_method': args.embedding_method,
    }

    facts_output_file = get_output_filename("facts", args)
    facts_embeddings = compute_embeddings(
        dataset.facts_data.values(),
        prepare_fact_text,
        facts_output_file,
        **common_args
    )

    posts_output_file = get_output_filename("posts", args)
    posts_embeddings = compute_embeddings(
        dataset.post_data.values(),
        prepare_post_text,
        posts_output_file,
        type="posts",
        query_prefix=args.query_prefix if args.embedding_method == 'standard' else None,
        **common_args,
    )

    print('--- Embedding Computation Summary ---')
    print(f'Embedding Method: {args.embedding_method}')
    print(f'Base Model: {args.base_model}')
    if args.embedding_method == 'prompt':
        print(f'Prompt Model Name: {args.prompt_model_name}')
        print(f'Prompt Init Text: "{args.prompt_init_text.strip()}"')
        print(f'Adapter Used: {adapter if adapter else args.prompt_model_name}')
    else:
        print(f'Adapter Model: {adapter if adapter else "None"}')
        print(f'Query Prefix Used: {args.query_prefix is not None}')

    print(f'Using Translations: {TRANSLATIONS}')
    print(f'Max Sequence Length: {args.max_seq_length}')
    print(f'Batch Size: {args.batch_size}')
    print(f'Number of fact embeddings: {len(facts_embeddings)}')
    print(f'Number of post embeddings: {len(posts_embeddings)}')
    print(f'Facts embeddings saved to: {facts_output_file}')
    print(f'Posts embeddings saved to: {posts_output_file}')
    print('-------------------------------------')


if __name__ == "__main__":
    main()
