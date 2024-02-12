import pandas as pd
import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, data_path, sp_path, max_seq_len: int = 128):
        self.data = pd.read_parquet(data_path)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_path)

        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        반드시 torch.Tensor 로 내보내는게 중요합니다.
        모든 vector의 길이가 동일해야 합니다. 그래야지 추후 batch 로 만들었을때도 문제가 없습니다.
        따라서 해당 함수에서 torch.tensor를 만들어서 내보내는게 맞습니다.
        """
        row = self.data.iloc[idx]
        korean = row.korean
        english = row.english

        src_tokenized = self.sp.encode(korean, add_bos=True, add_eos=True)
        tgt_tokenized = self.sp.encode(english, add_bos=True, add_eos=True)

        src_tokenized = src_tokenized[:self.max_seq_len]
        if src_tokenized[-1] != self.sp.eos_id():
            src_tokenized[-1] = self.sp.eos_id()
        if tgt_tokenized[-1] != self.sp.eos_id():
            tgt_tokenized[-1] = self.sp.eos_id()

        src_tensor = torch.zeros(self.max_seq_len, dtype=torch.int32)
        src_tensor[:len(src_tokenized)] = torch.tensor(src_tokenized)

        tgt_input = torch.zeros(self.max_seq_len, dtype=torch.int32)
        tgt_input[:max(len(tgt_tokenized) - 1, 0)] = torch.tensor(tgt_tokenized[:-1])

        tgt_output = torch.zeros(self.max_seq_len, dtype=torch.int32)
        tgt_output[:max(len(tgt_tokenized) - 1, 0)] = torch.tensor(tgt_tokenized[1:])

        return {'src': src_tensor, 'tgt_input': tgt_input, 'tgt_output': tgt_output}


if __name__ == '__main__':
    dataset = TranslationDataset('./_data/data.parquet',
                                 'sp-bpt-anderson.model')
    print(dataset[0])
