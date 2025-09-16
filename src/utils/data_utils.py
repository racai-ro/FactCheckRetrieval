import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from src.dataset import PostsDataset


def prepare_data(args):
    """
    Prepare the dataset for training and evaluation
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if PostsDataset is None:
        raise RuntimeError("PostsDataset could not be imported. Cannot prepare data.")

    dataset = PostsDataset('data')
    corpus = {fact['id']: f"{fact['title']}. {fact['claim']}" for fact in dataset.facts_data.values()}

    # Create train/test split
    train_keys, test_keys = train_test_split(list(dataset.pairs.keys()), test_size=0.1, random_state=args.seed)
    finetune_facts = set([c for sublist in dataset.pairs.values() for c in sublist])
    finetune_corpus = {str(k): corpus[k] for k in finetune_facts}

    # Return a dictionary with all the data variants needed
    return {
        'dataset': dataset,
        'corpus': corpus,
        'train_keys': train_keys,
        'test_keys': test_keys,
        'finetune_facts': finetune_facts,
        'finetune_corpus': finetune_corpus,
        'device': device
    }


def create_training_dataset(train_queries, train_pairs, data):
    """Create the training dataset from queries and pairs."""
    print("Creating training samples...")
    train_data = {"anchor": [], "positive": []}
    for query_id, query_text in train_queries.items():
        if query_id in train_pairs:
            for pos_id in train_pairs[query_id]:
                if pos_id in data['finetune_corpus']:
                    train_data["anchor"].append(query_text)
                    train_data["positive"].append(data['finetune_corpus'][pos_id])

    train_dataset = Dataset.from_dict(train_data)
    print(f"Created training dataset with {len(train_data['anchor'])} samples")
    return train_dataset


def prepare_queries(args, data):
    """Prepare queries and evaluation prompts based on the tuning method."""
    if args.method == 'prompt':
        queries = {post['id']: f"<query>{post['text']} {post['ocr_text']}"
                   for post in data['dataset'].post_data.values()}
        return queries

    queries = {
        post[
            'id']: f"<instruct>{args.query_prefix}\n<query>{post['text']} {post['ocr_text']}" if args.query_prefix else f"<query>{post['text']} {post['ocr_text']}"
        for post in data['dataset'].post_data.values()
    }
    return queries
