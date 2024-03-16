import json
import os
from pathlib import Path
from tempfile import gettempdir
from typing import Tuple

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer


def preprocess_raw_data():
    np.random.seed(23)
    data_dir_path = Path(gettempdir()) / 'korquad_preprocessed_data'
    train_data_path = data_dir_path / 'train_data.json'
    valid_data_path = data_dir_path / 'valid_data.json'
    test_data_path = data_dir_path / 'test_data.json'

    if train_data_path.exists() and valid_data_path.exists() and test_data_path.exists():
        return {'train': train_data_path, 'valid': valid_data_path, 'test': test_data_path}

    train_data, test_data = _download_data()
    train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Convert to Embedding Text
    tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
    train_data = _convert_to_encoded_data(tokenizer, train_data)
    valid_data = _convert_to_encoded_data(tokenizer, valid_data)
    test_data = _convert_to_encoded_data(tokenizer, test_data)

    with open(train_data_path, 'wt', encoding='utf-8') as f:
        json.dump(train_data, f)

    with open(valid_data_path, 'wt', encoding='utf-8') as f:
        json.dump(valid_data, f)

    with open(test_data_path, 'wt', encoding='utf-8') as f:
        json.dump(test_data, f)

    return {'train': train_data_path,
            'valid': valid_data_path,
            'test': test_data_path}


def _download_data() -> Tuple[list, list]:
    data_dir_path = Path(gettempdir()) / 'korquad_data'

    if not data_dir_path.exists():
        os.makedirs(data_dir_path)
    dataset = load_dataset('squad_kor_v1', data_dir=str(data_dir_path))
    train_data = dataset['train']
    valid_data = dataset['validation']

    return train_data.to_list(), valid_data.to_list()


def _convert_to_encoded_data(tokenizer, raw_data) -> list:
    tokenized_data = []
    for article in tqdm(raw_data):
        article_id = article['id']
        title = article['title']
        context = article['context']  # this is a paragraph
        question = article['question']

        for answer, answer_start in zip(article['answers']['text'], article['answers']['answer_start']):
            tokenized_data.append((tokenizer.encode(question),
                                   tokenizer.encode(context),
                                   article_id,
                                   answer,
                                   title))
    np.random.shuffle(tokenized_data)
    return tokenized_data
