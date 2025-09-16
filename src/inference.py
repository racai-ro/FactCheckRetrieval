import torch
from sentence_transformers import SentenceTransformer
from src.modules.query_only_prompt_st import QueryOnlyPromptSentenceTransformer
from src.modules.prompt_embedding import PromptEmbedding
from transformers import AutoConfig
from peft import PromptTuningConfig, PromptTuningInit
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import argparse


def create_mock_data() -> Tuple[List[Dict], List[Dict]]:
    """Create mock post and fact data for testing."""
    
    # Mock fact check data
    mock_facts = [
        {
            'id': 1,
            'title': 'COVID-19 vaccine safety',
            'claim': 'COVID-19 vaccines have been thoroughly tested and are safe for most people.',
            'title_en': 'COVID-19 vaccine safety',
            'claim_en': 'COVID-19 vaccines have been thoroughly tested and are safe for most people.'
        },
        {
            'id': 2, 
            'title': 'Climate change facts',
            'claim': 'Climate change is primarily caused by human activities and greenhouse gas emissions.',
            'title_en': 'Climate change facts',
            'claim_en': 'Climate change is primarily caused by human activities and greenhouse gas emissions.'
        },
        {
            'id': 3,
            'title': 'Electric vehicle batteries',
            'claim': 'Electric vehicle batteries can be recycled and their environmental impact is lower than gasoline cars.',
            'title_en': 'Electric vehicle batteries', 
            'claim_en': 'Electric vehicle batteries can be recycled and their environmental impact is lower than gasoline cars.'
        },
        {
            'id': 4,
            'title': 'Social media misinformation',
            'claim': 'Social media platforms have implemented fact-checking systems to combat misinformation.',
            'title_en': 'Social media misinformation',
            'claim_en': 'Social media platforms have implemented fact-checking systems to combat misinformation.'
        },
        {
            'id': 5,
            'title': 'Renewable energy efficiency',
            'claim': 'Solar and wind power have become cost-competitive with fossil fuels in many regions.',
            'title_en': 'Renewable energy efficiency',
            'claim_en': 'Solar and wind power have become cost-competitive with fossil fuels in many regions.'
        }
    ]
    
    # Mock post data
    mock_posts = [
        {
            'id': 101,
            'text': 'Just heard that COVID vaccines are dangerous and cause serious side effects. Is this true?',
            'en_text': 'Just heard that COVID vaccines are dangerous and cause serious side effects. Is this true?',
            'ocr_text': '',
            'ocr_en_text': ''
        },
        {
            'id': 102,
            'text': 'Climate change is a hoax created by politicians to control us!',
            'en_text': 'Climate change is a hoax created by politicians to control us!', 
            'ocr_text': '',
            'ocr_en_text': ''
        },
        {
            'id': 103,
            'text': 'Electric cars are worse for the environment because of battery waste',
            'en_text': 'Electric cars are worse for the environment because of battery waste',
            'ocr_text': '',
            'ocr_en_text': ''
        }
    ]
    
    return mock_posts, mock_facts


def prepare_fact_text(fact: Dict, use_translations: bool = False) -> str:
    """Prepare fact text for embedding, similar to compute_embeddings.py"""
    if use_translations:
        return f'{fact["title_en"]} {fact["claim_en"]}'
    return f'{fact["title"]}. {fact["claim"]}'


def prepare_post_text(post: Dict, embedding_method: str = 'standard', use_translations: bool = False) -> str:
    """Prepare post text for embedding, similar to compute_embeddings.py"""
    text = post['text'] or ''
    en_text = post['en_text'] or ''
    ocr_text = post['ocr_text'] or ''
    ocr_en_text = post['ocr_en_text'] or ''
    
    if use_translations:
        return f'{en_text} {ocr_en_text}'
    
    if embedding_method == 'prompt':
        return f'[QUERY]{text} {ocr_text}'
    else:
        return f'{text} {ocr_text}'


def create_model(embedding_method: str = 'standard', 
        base_model: str = "BAAI/bge-multilingual-gemma2",
                max_seq_length: int = 512,
        adapter_model: Optional[str] = None,
        prompt_model_name: Optional[str] = None,
        prompt_init_text: Optional[str] = None,
                device: str = 'cpu') -> SentenceTransformer:
    """Create model for embeddings, adapted from compute_embeddings.py"""
    
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
        
        # For demo purposes, we'll skip loading actual weights if they don't exist
        try:
            model.load_prompt_embedding_weights(prompt_model_name)
            if adapter_model:
                model.load_adapter(adapter_model)
        else:
                model.load_adapter(prompt_model_name)
        except Exception as e:
            print(f"Warning: Could not load prompt weights/adapter: {e}")
            print("Continuing with base model for demo purposes.")
        
        model.max_seq_length = max_seq_length
    
    elif embedding_method == 'standard':
        print(f"Using standard SentenceTransformer method with base model: {base_model}")
        try:
            model = SentenceTransformer(base_model, trust_remote_code=True)
            if adapter_model:
                print(f"Loading adapter {adapter_model} for standard method.")
                model.load_adapter(adapter_model)
        except Exception as e:
            print(f"Warning: Could not load model {base_model}: {e}")
            print("Falling back to a smaller model for demo purposes.")
            # Fallback to a smaller, commonly available model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
        model.max_seq_length = max_seq_length
        model.tokenizer.padding_side = "right"
        model = model.to(device)
    
        else:
        raise ValueError(f"Unknown embedding_method: {embedding_method}")
    
    return model


def compute_embeddings_for_inference(texts: List[str], 
                                   model: SentenceTransformer,
                                   embedding_method: str = 'standard',
    query_prefix: Optional[str] = None,
                                   batch_size: int = 32) -> torch.Tensor:
    """Compute embeddings for a list of texts."""
    
    if embedding_method == 'prompt':
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_tensor=True,
            batch_size=batch_size
        )
    else:
        embeddings = model.encode(
            texts,
            prompt=query_prefix,
            normalize_embeddings=True,
            convert_to_tensor=True,
            batch_size=batch_size
        )
    
    return embeddings


def find_top_k_facts(query_embedding: torch.Tensor, 
                    fact_embeddings: torch.Tensor, 
                    fact_ids: List[int], 
                    k: int = 5) -> List[Tuple[int, float]]:
    """Find top-k most similar facts to the query."""
    
    # Compute cosine similarities
    similarities = torch.cosine_similarity(query_embedding.unsqueeze(0), fact_embeddings, dim=1)
    
    # Get top-k indices and scores
    top_k_scores, top_k_indices = torch.topk(similarities, min(k, len(similarities)))
    
    # Return fact IDs and scores
    results = [(fact_ids[idx.item()], score.item()) for idx, score in zip(top_k_indices, top_k_scores)]

    return results


def inference_demo(query: Optional[str] = None, 
                  embedding_method: str = 'standard',
                  base_model: str = "BAAI/bge-multilingual-gemma2",
                  adapter_model: str = 'multilingual-fact-retrieval/checkpoint-40',
                  prompt_model_name: str = 'multilingual-fact-retrieval-gemma-prompt/checkpoint-140',
                  prompt_init_text: str = "Given a social media post as a query, retrieve fact checks that verify or debunk the post.\n",
                  query_prefix: str = "<instruct>Given a social media post as a query, retrieve fact checks that verify or debunk the post.\n<query>",
    max_seq_length: int = 1024,
                  top_k: int = 3,
                  use_translations: bool = False,
                  device: str = 'cpu',
                  verbose: bool = True) -> Dict:
    """
    Main inference function that demonstrates fact retrieval.
    
    Args:
        query: User query text. If None, uses mock post data.
        embedding_method: 'standard' or 'prompt'
        base_model: Model identifier
        top_k: Number of top facts to retrieve
        use_translations: Whether to use English translations
        device: 'cpu' or 'cuda'
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with query, top facts, and scores
    """
    
    if verbose:
        print("="*50)
        print("FACT RETRIEVAL INFERENCE DEMO")
        print("="*50)
    
    # Create mock data
    mock_posts, mock_facts = create_mock_data()
    
    # Use provided query or default to first mock post
    if query is None:
        query = mock_posts[0]['text']
        if verbose:
            print(f"Using mock query: {query}")
    else:
        if verbose:
            print(f"User query: {query}")
    
    # Create model
    try:
        # Handle adapter model parameter (set to None if "none" specified)
        adapter = adapter_model if adapter_model and adapter_model.lower() != "none" else None
        
        model = create_model(
        embedding_method=embedding_method,
        base_model=base_model,
            max_seq_length=max_seq_length,
            adapter_model=adapter,
        prompt_model_name=prompt_model_name,
        prompt_init_text=prompt_init_text,
            device=device
        )
    except Exception as e:
        if verbose:
            print(f"Error creating model: {e}")
        return {"error": str(e)}
    
    # Prepare texts
    if embedding_method == 'prompt':
        query_text = f'[QUERY]{query}'
    else:
        # For standard method, use query prefix if provided
        query_text = f"{query_prefix}{query}" if query_prefix else query
    
    fact_texts = [prepare_fact_text(fact, use_translations) for fact in mock_facts]
    
    if verbose:
        print(f"\nPrepared query text: {query_text}")
        print(f"Number of facts to search: {len(fact_texts)}")
    
    # Compute embeddings
    try:
        if verbose:
            print("Computing embeddings...")
        
        query_embedding = compute_embeddings_for_inference([query_text], model, embedding_method, query_prefix if embedding_method == 'standard' else None)
        fact_embeddings = compute_embeddings_for_inference(fact_texts, model, embedding_method)
        
        if verbose:
            print(f"Query embedding shape: {query_embedding.shape}")
            print(f"Fact embeddings shape: {fact_embeddings.shape}")
        
    except Exception as e:
        if verbose:
            print(f"Error computing embeddings: {e}")
        return {"error": f"Error computing embeddings: {str(e)}"}
    
    # Find top-k facts
    fact_ids = [fact['id'] for fact in mock_facts]
    top_facts = find_top_k_facts(query_embedding[0], fact_embeddings, fact_ids, top_k)
    
    # Prepare results
    results = {
        'query': query,
        'top_facts': []
    }
    
    if verbose:
        print(f"\nTop {top_k} most relevant facts:")
        print("-" * 40)
    
    for i, (fact_id, score) in enumerate(top_facts, 1):
        fact = next(f for f in mock_facts if f['id'] == fact_id)
        fact_result = {
            'rank': i,
            'fact_id': fact_id,
            'title': fact['title'],
            'claim': fact['claim'],
            'similarity_score': score
        }
        results['top_facts'].append(fact_result)
        
        if verbose:
            print(f"{i}. [ID: {fact_id}] {fact['title']}")
            print(f"   Claim: {fact['claim']}")
            print(f"   Similarity: {score:.4f}")
            print()

    return results


def parse_args():
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(description='Fact-checking inference with embeddings')
    
    # Query and output settings
    parser.add_argument('--query', type=str, default=None,
                        help='Query text to search for. If not provided, uses mock data.')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of top facts to retrieve')
    parser.add_argument('--output_format', type=str, default='console', choices=['console', 'json'],
                        help='Output format: console or json')
    parser.add_argument('--output_file', type=str, default=None,
                        help='File to save results to (optional)')
    
    # Model configuration (matching compute_embeddings.py)
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
    
    # Model settings
    parser.add_argument('--max_seq_length', type=int, default=1024,
                        help='Max sequence length for the model')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use: cpu or cuda')
    parser.add_argument('--translations', action='store_true',
                        help='Use English translations instead of original text')
    
    # Fallback settings
    parser.add_argument('--use_fallback', action='store_true',
                        help='Use a simpler fallback model if the main model fails')
    parser.add_argument('--fallback_model', type=str, default='all-MiniLM-L6-v2',
                        help='Fallback model to use if main model fails')
    
    # Debug settings
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Enable verbose output')
    parser.add_argument('--quiet', action='store_true',
                        help='Disable verbose output')
    
    return parser.parse_args()


def main():
    """Main function that handles command line arguments and runs inference."""
    args = parse_args()

    # Handle quiet flag
    verbose = args.verbose and not args.quiet
    
    if verbose:
        print("="*60)
        print("FACT-CHECKING INFERENCE")
        print("="*60)
        print(f"Query: {args.query if args.query else 'Using mock data'}")
        print(f"Model: {args.base_model}")
        print(f"Method: {args.embedding_method}")
        print(f"Top-K: {args.top_k}")
        print(f"Device: {args.device}")
        print("-"*60)
    
    # Run inference with fallback logic
    try:
        result = inference_demo(
            query=args.query,
            embedding_method=args.embedding_method,
            base_model=args.base_model,
            adapter_model=args.adapter_model,
            prompt_model_name=args.prompt_model_name,
            prompt_init_text=args.prompt_init_text,
            query_prefix=args.query_prefix,
            max_seq_length=args.max_seq_length,
            top_k=args.top_k,
            use_translations=args.translations,
            device=args.device,
            verbose=verbose
        )
    except Exception as e:
        if args.use_fallback:
            if verbose:
                print(f"Error with main model: {e}")
                print(f"Falling back to simpler model: {args.fallback_model}")
            
            result = inference_demo(
                query=args.query,
                embedding_method='standard',  # Use standard method for fallback
                base_model=args.fallback_model,
                adapter_model='none',  # Skip adapter for fallback
                prompt_model_name=None,
                prompt_init_text=None,
                query_prefix=None,  # Simplify for fallback
                max_seq_length=512,  # Smaller sequence length
                top_k=args.top_k,
                use_translations=False,
                device=args.device,
                verbose=verbose
            )
        else:
            if verbose:
                print(f"Error: {e}")
            result = {"error": str(e)}
    
    # Handle output
    if args.output_format == 'json' or args.output_file:
        output_data = {
            "query": args.query,
            "model_config": {
                "embedding_method": args.embedding_method,
                "base_model": args.base_model,
                "adapter_model": args.adapter_model,
                "device": args.device
            },
            "results": result
        }
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            if verbose:
                print(f"\nResults saved to: {args.output_file}")
        else:
            print(json.dumps(output_data, indent=2))
    
    elif verbose and 'error' not in result:
        print("\n" + "="*60)
        print("INFERENCE COMPLETE")
        print("="*60)
        print(f"Query processed successfully")
        print(f"Found {len(result.get('top_facts', []))} relevant facts")
        
        if result.get('top_facts'):
            print(f"Top result: {result['top_facts'][0]['title']}")
            print(f"Best similarity score: {result['top_facts'][0]['similarity_score']:.4f}")


if __name__ == "__main__":
    main()
