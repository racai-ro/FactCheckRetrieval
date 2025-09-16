from torch.utils.data import Dataset
import pandas as pd
import ast
import numpy as np


def parse_cols(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda s: ast.literal_eval(s.replace("nan", "0").replace('\n', '\n')) if s else s)
    return df


def get_facts_data(data_path='data'):
    df_fact_checks = pd.read_csv(f'{data_path}/fact_checks.csv',
                                 usecols=['fact_check_id', 'claim', 'instances', 'title'],
                                 dtype={'fact_check_id': int},
                                 keep_default_na=False)

    df_fact_checks = parse_cols(df_fact_checks, ['claim', 'instances', 'title'])

    return {
        int(row['fact_check_id']): {
            'id': row['fact_check_id'],
            'claim': row['claim'][0] if isinstance(row['claim'], tuple) else str(row['claim']),
            'claim_en': row['claim'][1] if isinstance(row['claim'], tuple) else '',
            'title': row['title'][0] if isinstance(row['title'], tuple) else str(row['title']),
            'title_en': row['title'][1] if isinstance(row['title'], tuple) else '',
            'instances': row['instances']
        } for _, row in df_fact_checks.iterrows()}


def get_posts_data(data_path='data'):
    df_posts = pd.read_csv(f'{data_path}/posts.csv',
                           usecols=['post_id', 'instances', 'ocr', 'verdicts', 'text'],
                           dtype={'post_id': int},
                           keep_default_na=False)

    df_posts = parse_cols(df_posts, ['instances', 'ocr', 'verdicts', 'text'])

    def process_ocr(ocr_list):
        if not ocr_list:
            return '', ''
        ocr_texts = list(zip(*ocr_list))
        return ''.join(ocr_texts[0]), ''.join(ocr_texts[1])

    def process_text(texts):
        if not texts:
            return '', ''
        return texts[0], texts[1]

    # Apply vectorized operations
    df_posts['ocr_processed'] = df_posts['ocr'].apply(process_ocr)
    df_posts['text_processed'] = df_posts['text'].apply(process_text)

    # Use dictionary comprehension
    return {int(row['post_id']): {
        'id': row['post_id'],
        'verdict': "Unknown" if not row['verdicts'] else row['verdicts'][0],
        'text': row['text_processed'][0],
        'en_text': row['text_processed'][1],
        'ocr_text': row['ocr_processed'][0],
        'ocr_en_text': row['ocr_processed'][1]
    } for _, row in df_posts.iterrows()}


def get_pairings(data_path='data'):
    df_pairs = pd.read_csv(f'{data_path}/pairs.csv',
                           dtype={'post_id': int, 'fact_check_id': int},
                           keep_default_na=False)

    pairs_dict = df_pairs.groupby('post_id')['fact_check_id'].agg(list).to_dict()
    return {int(k): [int(v) for v in vals] for k, vals in pairs_dict.items()}


class PostsDataset(Dataset):

    def __init__(self, data_path='data', phase='train'):
        self.post_data = get_posts_data(data_path)
        self.facts_data = get_facts_data(data_path)
        if phase == 'train':
            self.pairs = get_pairings(data_path)
            self.pairings = list(self.pairs.items())
        else:
            self.pairings = []

    def __len__(self):
        return len(self.pairings)

    def __getitem__(self, idx):
        return {'post': self.post_data[self.pairings[idx][0]],
                'facts': [self.facts_data[fact_id] for fact_id in self.pairings[idx][1]]}


if __name__ == '__main__':
    dataset = PostsDataset()
    lengths = []
    for entry in dataset.post_data.values():
        combined_text = entry['text'] + ' ' + entry['ocr_text']
        lengths.append(len(combined_text))

    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    min_len = min(lengths)

    print(f"Average post length: {avg_len:.2f} characters")
    print(f"Maximum post length: {max_len} characters")
    print(f"Minimum post length: {min_len} characters")

    # Show distribution in percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        length_at_p = np.percentile(lengths, p)
        print(f"{p}th percentile length: {length_at_p:.2f} characters")

    print(len(dataset))
    print(len(dataset.post_data))
    print(len(dataset.facts_data))
    print(dataset[0])
