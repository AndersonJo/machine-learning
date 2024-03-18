import logging

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from data import KorQuadDataset, InBatchNegativeSampler, KorquadCollator
from preprocess import _download_data, preprocess_raw_data
from model import KoBertBiEncoder

# 로거 생성 및 로그 레벨 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 콘솔 출력을 위한 핸들러 생성 및 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)


def train():
    data_paths = preprocess_raw_data()

    model = KoBertBiEncoder(lr=1e-5, betas=(0.9, 0.99))
    train_dataset = KorQuadDataset(data_paths['train'])
    test_dataset = KorQuadDataset(data_paths['test'])
    pad_id = train_dataset.pad_id

    # Max Sequence Size 는 GPU 메모리를 겁나게 갈가 먹습니다.
    # 되도록 작은 값을 사용하도록 합니다.
    train_loader = DataLoader(train_dataset,
                              batch_size=120,
                              shuffle=True,
                              # batch_sampler=InBatchNegativeSampler(train_dataset, batch_size=32, drop_last=False),
                              collate_fn=KorquadCollator(pad_id=pad_id, max_seq_len=90),
                              num_workers=4
                              )
    valid_loader = DataLoader(test_dataset,
                              batch_size=120,
                              # batch_sampler=InBatchNegativeSampler(valid_dataset, batch_size=32, drop_last=False),
                              collate_fn=KorquadCollator(pad_id=pad_id, max_seq_len=90),
                              num_workers=4
                              )

    checkpointer = ModelCheckpoint(
        dirpath='./checkpoints',
        filename='dense-passage-retrieval-{epoch:02d}-{step:06d}-{valid_loss:.4f}',
        monitor='valid_loss',
        every_n_epochs=1,
        save_top_k=-1,
        mode='min'
    )

    # Resume 할때 필요
    checkpoint_path = './checkpoints/dense-passage-retrieval-epoch=07-step=008889-valid_loss=4.5246.ckpt'
    tensorboard = TensorBoardLogger('./tb_logs', name='dense-passage-retrieval')
    trainer = Trainer(max_epochs=2000,
                      accelerator='cuda',
                      devices=1,
                      enable_checkpointing=True,
                      enable_progress_bar=True,
                      enable_model_summary=True,
                      # hard negative sampling - LightningDataModule 의 train_dataloader 함수에서 Subset 사용해서 추가
                      reload_dataloaders_every_n_epochs=True,
                      logger=tensorboard,
                      callbacks=[checkpointer],
                      )
    trainer.fit(model, train_loader, valid_loader,
                ckpt_path=checkpoint_path
                )
    # trainer.validate(model, valid_loaderB


if __name__ == '__main__':
    train()
