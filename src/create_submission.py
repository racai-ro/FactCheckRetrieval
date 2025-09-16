import torch
import json
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Create fact retrieval predictions')
    parser.add_argument('--model_name', type=str, default='BAAI_bge-multilingual-gemma2_False.pt',
                        help='Name of the model used for embeddings')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top facts to retrieve')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for computation (cuda or cpu)')
    parser.add_argument('--tasks_file', type=str, default='data/tasks.json',
                        help='Path to the tasks file')
    parser.add_argument('--monolingual_ground_truth', type=str, default='data/dev_ground_truth/monolingual_reference.json',
                        help='Path to the monolingual ground truth file')
    parser.add_argument('--crosslingual_ground_truth', type=str, default='data/dev_ground_truth/crosslingual_reference.json',
                        help='Path to the crosslingual ground truth file')
    return parser.parse_args()


def create_predictions(posts_ids, posts_embeddings, facts_tensor, fact_ids, top_k):
    """Create predictions for a list of post IDs."""
    results = {}
    for post_id in tqdm(posts_ids, desc='Computing similarities'):
        post_embedding = posts_embeddings[int(post_id)]
        similarities = post_embedding @ facts_tensor.T

        top_k_indices = torch.topk(similarities, top_k).indices
        top_k_facts = [int(fact_ids[idx]) for idx in top_k_indices.tolist()]
        results[post_id] = top_k_facts
    return results


def perform_error_analysis(predictions_file, reference_file, output_file):
    """Perform error analysis by comparing predictions with reference data."""
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    with open(reference_file, 'r') as f:
        reference = json.load(f)

    errors = {}
    for post_id, predicted_facts in predictions.items():
        actual_fact = reference.get(post_id, [None])[0]
        if actual_fact not in set(predicted_facts):
            errors[post_id] = {
                'predicted': predicted_facts,
                'actual': actual_fact
            }

    with open(output_file, 'w') as f:
        json.dump(errors, f, indent=2)

    print(f"Error analysis completed. {len(errors)} errors found.")


def main():
    args = parse_args()

    posts_embeddings = torch.load(f'posts_embeddings_{args.model_name}', map_location=args.device)
    facts_embeddings = torch.load(f'facts_embeddings_{args.model_name}', map_location=args.device)

    with open(args.tasks_file, 'r') as f:
        tasks = json.load(f)

    fact_ids = tasks['crosslingual']['fact_checks']
    facts_tensor = torch.stack([facts_embeddings[fid] for fid in fact_ids])

    crosslingual_results = create_predictions(
        tasks['crosslingual']['posts_dev'],
        posts_embeddings,
        facts_tensor,
        fact_ids,
        args.top_k
    )

    with open('crosslingual_predictions.json', 'w') as f:
        json.dump(crosslingual_results, f, indent=2)

    monolingual_results = {}
    for lang in tasks['monolingual'].keys():
        fact_ids = tasks['monolingual'][lang]['fact_checks']
        facts_tensor = torch.stack([facts_embeddings[fid] for fid in fact_ids])

        lang_results = create_predictions(
            tasks['monolingual'][lang]['posts_dev'],
            posts_embeddings,
            facts_tensor,
            fact_ids,
            args.top_k
        )
        monolingual_results.update(lang_results)

    with open('monolingual_predictions.json', 'w') as f:
        json.dump(monolingual_results, f, indent=2)



    print(f"Created submissions:")
    print(f"Crosslingual: {len(crosslingual_results)} predictions")
    print(f"Monolingual: {len(monolingual_results)} predictions")

    perform_error_analysis('monolingual_predictions.json', 'data/dev_ground_truth/monolingual_reference.json', 'monolingual_errors.json')
    perform_error_analysis('crosslingual_predictions.json', 'data/dev_ground_truth/crosslingual_reference.json', 'crosslingual_errors.json')


if __name__ == "__main__":
    main()
