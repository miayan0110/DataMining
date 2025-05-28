import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

def load_entity_embeddings_df(path):
    print(f'Loading entity embeddings...')
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            entity = parts[0]
            vec = [float(x) for x in parts[1:]]
            records.append({'entity': entity, 'embedding': vec})
    return pd.DataFrame(records)

def load_news_wiki_ids_df(path):
    print(f'Loading news wiki ids...')
    df = pd.read_csv(path, sep='\t', usecols=['news_id', 'title_entities', 'abstract_entities'])
    def extract_ids(s) -> list:
        if pd.isna(s) or not s.strip():
            return []
        try:
            entities = json.loads(s)
        except json.JSONDecodeError:
            entities = json.loads(s.replace('""', '"'))
        return [ent['WikidataId'] for ent in entities if 'WikidataId' in ent]
    df['title_wiki_ids'] = df['title_entities'].apply(extract_ids)
    df['abstract_wiki_ids'] = df['abstract_entities'].apply(extract_ids)
    return df[['news_id', 'title_wiki_ids', 'abstract_wiki_ids']]

def load_behaviors_df(path, train=True):
    print(f'Loading behaviors...')
    df = pd.read_csv(path, sep='\t', usecols=['id', 'clicked_news', 'impressions'])
    
    # 處理 clicked_news
    df['clicked_news'] = df['clicked_news'].fillna('').apply(lambda x: x.split() if x else [])
    
    if train:
    # 解析 impressions 與 click
        def parse_impressions(s):
            if pd.isna(s) or not s.strip():
                return [], []
            news_click_pairs = s.split()
            news_ids, clicks = [], []
            for pair in news_click_pairs:
                nid, lbl = pair.split('-')
                news_ids.append(nid)
                clicks.append(int(lbl))
            return news_ids, clicks
        
        parsed = df['impressions'].fillna('').apply(parse_impressions)
        df['impressions'] = parsed.apply(lambda x: x[0])
        df['click'] = parsed.apply(lambda x: x[1])
    else:
        df['impressions'] = df['impressions'].fillna('').apply(lambda x: x.split() if x else [])
    
    return df

def news_to_emb(news_wiki_df, entity_embeddings_df, target_news_ids):
    # 確定 embedding 維度
    if entity_embeddings_df.empty:
        return torch.tensor([])
    emb_dim = len(entity_embeddings_df['embedding'].iloc[0])

    # 建立索引方便查詢
    news_wiki = news_wiki_df.set_index('news_id')
    entity_emb = entity_embeddings_df.set_index('entity')

    flat_list = []
    for nid in target_news_ids:
        if nid not in news_wiki.index:
            flat_list.extend([0.0] * emb_dim)
            continue
        wiki_ids = news_wiki.at[nid, 'title_wiki_ids'] + news_wiki.at[nid, 'abstract_wiki_ids']
        vecs = [np.array(entity_emb.at[wid, 'embedding'], dtype=float)
                for wid in wiki_ids if wid in entity_emb.index]
        mean_emb = np.mean(vecs, axis=0) if vecs else np.zeros(emb_dim, dtype=float)
        flat_list.append(mean_emb.tolist())

    # 轉為 1D Tensor
    return torch.tensor(flat_list, dtype=torch.float32)


class TrainingData(Dataset):
    def __init__(self, max_history_len=0):
        super().__init__()
        self.behaviors_df = load_behaviors_df('./data/train/train_behaviors.tsv')
        self.news_wiki_df = load_news_wiki_ids_df('./data/train/train_news.tsv')
        self.entity_embeddings_df = load_entity_embeddings_df('./data/train/train_entity_embedding.vec')
        self.max_history_len = max_history_len

    def __len__(self):
        return self.behaviors_df.shape[0]
    
    def __getitem__(self, index):
        id = self.behaviors_df.loc[index, 'id']
        clicked_news = self.behaviors_df.loc[index, 'clicked_news']
        impressions = self.behaviors_df.loc[index, 'impressions']
        click = torch.tensor(self.behaviors_df.loc[index, 'click'])

        clicked_news = clicked_news[-self.max_history_len:] # for validation (limit used history length)

        clicked_news_emb = news_to_emb(self.news_wiki_df, self.entity_embeddings_df, clicked_news)
        impressions_emb = news_to_emb(self.news_wiki_df, self.entity_embeddings_df, impressions)

        return id, clicked_news_emb, impressions_emb, click


class TestData(Dataset):
    def __init__(self):
        super().__init__()
        self.behaviors_df = load_behaviors_df('./data/test/test_behaviors.tsv', train=False)
        self.news_wiki_df = load_news_wiki_ids_df('./data/test/test_news.tsv')
        self.entity_embeddings_df = load_entity_embeddings_df('./data/test/test_entity_embedding.vec')

    def __len__(self):
        return self.behaviors_df.shape[0]
    
    def __getitem__(self, index):
        id = self.behaviors_df.loc[index, 'id']
        clicked_news = self.behaviors_df.loc[index, 'clicked_news']
        impressions = self.behaviors_df.loc[index, 'impressions']

        clicked_news_emb = news_to_emb(self.news_wiki_df, self.entity_embeddings_df, clicked_news)
        impressions_emb = news_to_emb(self.news_wiki_df, self.entity_embeddings_df, impressions)

        return id, clicked_news_emb, impressions_emb