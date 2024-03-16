import lightning as pl
import torch.nn
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from transformers import BertModel


class KoBertBidirectionalEncoder(pl.LightningModule):

    def __init__(self, lr=0.001) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.query_encoder = BertModel.from_pretrained('skt/kobert-base-v1')
        self.passage_encoder = BertModel.from_pretrained('skt/kobert-base-v1')
        self.output_embedding_size = self.passage_encoder.pooler.dense.out_features  # 768

    def forward(self, x: torch.LongTensor, attention_mask: torch.LongTensor, input_type: str) -> torch.FloatTensor:
        if input_type == 'query':
            return self.query_encoder(x, attention_mask=attention_mask)
        return self.passage_encoder(x, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):

        import ipdb
        ipdb.set_trace()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=self.lr)






