import json
import logging
from pathlib import Path
from typing import Optional, List, Union, Iterator, Iterable, Sized

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, BatchSampler, Sampler, RandomSampler, SequentialSampler
from transformers import AutoTokenizer


class KorQuadDataset(Dataset):

    def __init__(self, kor_quad_path: Union[str, Path]):
        kor_quad_path: Path = Path(kor_quad_path)
        with open(kor_quad_path, 'r') as f:
            self.data: List[tuple] = json.load(f)
            logging.info(f'Data Loaded from {kor_quad_path}')

        self.tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
        self.pad_id = self.tokenizer.pad_token_id

    @property
    def dataset(self) -> List[tuple]:
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple:
        encoded_question, encoded_context, article_id, answer, title = self.data[idx]
        return encoded_question, encoded_context, article_id


class InBatchNegativeSampler(BatchSampler):
    """
    In-batch negative 학습을 위해서 중복 answer를 갖지 않도록 batch 를 구성합니다.
    따라서 중복되는 데이터 일부를  패스 하기 때문에, Dataset 에서의 전체 데이터셋의 크기보다 작을 수 있습니다.
    """

    ARTICLE_ID_IDX = 2

    def __init__(self, data_source: Sized,
                 batch_size: int,
                 drop_last: bool = False,
                 shuffle: bool = False) -> None:

        # Sampler 사용하기전에, RandomSampler 또는 SequentialSampler 를 wrapping 해야 됩니다.
        if shuffle:
            sampler = RandomSampler(data_source)
        else:
            sampler = SequentialSampler(data_source)
        super().__init__(sampler, batch_size=batch_size, drop_last=drop_last)

    def __iter__(self) -> Iterator[List[int]]:
        """
        만약 ground truth 값이 있다면 (y=1), 하나만 남겨두고 전부다 negative 로 둬도 될듯.
        """
        duplicates = set()
        sampled_ids = []
        for idx in self.sampler:
            item = self.sampler.data_source[idx]
            article_id = item[self.ARTICLE_ID_IDX]
            if article_id in duplicates:
                logging.debug(f'duplicated article_id: {article_id}')
                continue

            sampled_ids.append(idx)
            duplicates.add(article_id)
            if len(sampled_ids) >= self.batch_size:
                yield sampled_ids
                sampled_ids.clear()
                duplicates.clear()

        if len(sampled_ids) > 0 and not self.drop_last:
            yield sampled_ids


class KorquadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: List[tuple]):
        """
         - batch_q: batch of encoded questions
         - batch_p: batch of paragraphs (context)
        """

        batch_q = pad_sequence([torch.Tensor(i[0]) for i in batch],
                               batch_first=True,
                               padding_value=self.pad_id)
        batch_q_attn_mask = torch.Tensor(batch_q != self.pad_id).long()
        batch_p = pad_sequence([torch.Tensor(i[1]) for i in batch],
                               batch_first=True,
                               padding_value=self.pad_id)
        batch_p_attn_mask = torch.Tensor(batch_p != self.pad_id).long()

        return batch_q, batch_p_attn_mask, batch_p, batch_p_attn_mask
