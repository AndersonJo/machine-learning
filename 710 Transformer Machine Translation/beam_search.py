import torch
from sentencepiece import SentencePieceProcessor
from torch import nn

from model import TransformerModule
from typing import Optional


class BeamSearch:

    def __init__(self, model: TransformerModule, sp: SentencePieceProcessor, device: Optional[str] = None,
                 max_seq_len: int = 128):
        self.model = model
        self.sp = sp
        self.max_seq_len = max_seq_len
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def greedy_search(self, text: str):
        self.model.eval()
        self.model.to(self.device)

        src_tensor = self.create_source_tensor(text)
        src_padding_mask = self.get_padding_mask(src_tensor, pad_idx=self.sp.pad_id())

        # Get initial encoder output
        # if we wrap it with `torch.no_grad()` it doesn't work for some reason.
        memory = self.model.encode(src_tensor, src_padding_mask=src_padding_mask)
        memory = memory.to(self.device)

        with torch.no_grad():
            # Create decoder input.
            # it starts with <bos> token.
            y_pred = torch.ones(1, 1).fill_(self.sp.bos_id()).type(torch.long).to(self.device)

            for i in range(self.max_seq_len - 1):
                tgt_mask = (nn.Transformer.generate_square_subsequent_mask(y_pred.size(1))
                            .type(torch.bool)).to(self.device)

                out = self.model.decode(y_pred, memory, tgt_mask)
                out = out.transpose(0, 1)
                prob = self.model.out(out[:, -1])
                _, next_words = torch.max(prob, dim=1)
                next_word = next_words[-1].item()

                y_pred = torch.cat([y_pred, torch.ones(1, 1).type_as(src_tensor.data).fill_(next_word)], dim=1)
                if next_word == self.sp.eos_id():
                    break
        return y_pred

    def convert_output_to_text(self, y_pred: torch.Tensor):
        return self.sp.Decode(y_pred[0].tolist())

    def create_source_tensor(self, text: str) -> torch.Tensor:
        # Create src input
        src_tokenized = self.sp.Encode(text, add_bos=True, add_eos=True)
        src_tokenized = src_tokenized[:self.max_seq_len]
        if src_tokenized[-1] != self.sp.eos_id():
            src_tokenized[-1] = self.sp.eos_id()

        src_tensor = torch.zeros(self.max_seq_len, dtype=torch.int32).to(self.model.device)
        src_tensor[: len(src_tokenized)] = torch.tensor(src_tokenized)
        src_tensor = src_tensor.unsqueeze(0)
        src_tensor = src_tensor.to(self.device)
        return src_tensor

    @staticmethod
    def get_padding_mask(seq, pad_idx: int):
        return torch.tensor(seq == pad_idx).to(seq.device)


if __name__ == '__main__':
    from model import TransformerModule
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load("sp-bpt-anderson.model")

    model_path = (
        "checkpoints/machine-translation-mode=1.0-epoch=94-step=415625-loss=1.9441.ckpt"
    )
    model = TransformerModule.load_from_checkpoint(model_path)
    model.eval()

    search = BeamSearch(model, sp, device='cpu')
    output = search.greedy_search("제가 이번 여름 휴가 보낸 이야기를 할게요.")
    print(search.convert_output_to_text(output))
