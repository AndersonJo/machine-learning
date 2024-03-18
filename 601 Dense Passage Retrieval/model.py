from typing import Any

import lightning as pl
import torch.nn
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from transformers import BertModel


class KoBertBiEncoder(pl.LightningModule):

    def __init__(self, lr=0.001) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.quesion_encoder = BertModel.from_pretrained('skt/kobert-base-v1')
        self.passage_encoder = BertModel.from_pretrained('skt/kobert-base-v1')
        self.output_embedding_size = self.passage_encoder.pooler.dense.out_features  # 768

    def forward(self, x: torch.LongTensor, attention_mask: torch.LongTensor, is_question: bool) -> torch.FloatTensor:
        if is_question:
            return self.quesion_encoder(x, attention_mask=attention_mask).pooler_output
        return self.passage_encoder(x, attention_mask=attention_mask).pooler_output

    def training_step(self, batch, batch_idx):
        similarities = self.do_similarity_step(batch)

        loss = self.ibn_loss(similarities)
        batch_acc = self.calculate_batch_accuracy(similarities)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_batch_acc', batch_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        similarities = self.do_similarity_step(batch)

        loss = self.ibn_loss(similarities)
        batch_acc = self.calculate_batch_accuracy(similarities)

        self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('valid_batch_acc', batch_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def do_similarity_step(self, batch):
        q_encoded, q_attn_mask, p_encoded, p_attn_mask = batch
        q_encoded, q_attn_mask, p_encoded, p_attn_mask = (
            q_encoded.to(self.device),
            q_attn_mask.to(self.device),
            p_encoded.to(self.device),
            p_attn_mask.to(self.device)
        )

        q_emb = self(q_encoded, attention_mask=q_attn_mask, is_question=True)
        p_emb = self(p_encoded, attention_mask=p_attn_mask, is_question=False)
        similarities = torch.matmul(q_emb, p_emb.T)  # calculate similarity
        return similarities

    def ibn_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        in-batch negative loss
        """
        batch_size = pred.size(0)
        target = torch.arange(batch_size).to(self.device)
        return torch.nn.functional.cross_entropy(pred, target)

    def calculate_batch_accuracy(self, pred: torch.Tensor):
        """
        Batch 내에서의 accuracy 를 계산
        """
        batch_size = pred.size(0)
        target = torch.arange(batch_size)  # target = [0, 1, 2, 3, 4, ...]
        # max(1) -> values 그리고 indices 가 있으며, indices 는 max value 의 index 번호 입니다.
        # 즉 [0, 1, 2, 3, 4 .. ] 이렇게 나와서 target 과 일치하는게 몇개 있는지 계산 합니다.
        return (pred.detach().cpu().max(1).indices == target).sum().float() / batch_size

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
