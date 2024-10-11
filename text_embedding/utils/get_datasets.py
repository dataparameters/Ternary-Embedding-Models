import json
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def load_nli_texts(file_path):
    texts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # 跳过空行
                data = json.loads(line)
                texts.append(data['text1'])
                texts.append(data['text2'])
    return texts

def load_t2ranking_sentences(parquet_file, cols, max_length=256):
    t2ranking = pd.read_parquet(parquet_file)
    sentences = []
    for col in cols:
        short_sentences = t2ranking[t2ranking[col].apply(lambda x: len(x) < max_length)][col].tolist()
        sentences.extend(short_sentences)
    return sentences

def save_texts_to_json(texts, output_file):
    """
    将句子列表保存到json文件
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(texts, file, ensure_ascii=False, indent=4)

class TextPairDataset(Dataset):
    """
    文本数据集
    """
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def create_dataloader(texts, batch_size=512, shuffle=True):
    """
    创建DataLoader
    """
    dataset = TextPairDataset(texts)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def text_dataloader(batch_size=512,save=False):  
    nli_zh_file = '/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/datasets/train_data_zh/sampled_data_nli_zh-train-25k.jsonl'
    t2ranking_file = '/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/datasets/train_data_zh/train-00000-of-00001.parquet'

    texts1 = load_nli_texts(nli_zh_file)
    texts2 = load_t2ranking_sentences(t2ranking_file, ['anchor', 'positive', 'negative'])
    texts1 += texts2
    
    if save:
        save_texts_to_json(texts1, 'list.json')

    trainloader = create_dataloader(texts1, batch_size=batch_size)

    return trainloader
