from argparse import ArgumentParser, Namespace

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from torch.utils.data import DataLoader

from data import TranslationDataset
from model import TransformerModule


def init() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--batch', default=64, type=int)
    opt = parser.parse_args()
    return opt


def train():
    opt = init()
    train_dataset = TranslationDataset('./_data/train.parquet', 'sp-bpt-anderson.model')
    valid_dataset = TranslationDataset('./_data/valid.parquet', 'sp-bpt-anderson.model')
    test_dataset = TranslationDataset('./_data/test.parquet', 'sp-bpt-anderson.model')

    train_loader = DataLoader(train_dataset, batch_size=opt.batch)
    valid_loader = DataLoader(valid_dataset, batch_size=opt.batch)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch)

    model = TransformerModule(src_vocab_size=8000, tgt_vocab_size=8000, d_model=256)

    # 보통 val_loss 를 잡는데, 실무에서 데이터 사이즈가 워낙 커서, 학습 도중에도 checkpoint 저장이 필요합니다. 그래서 loss 로 했습니다.
    # 이를 위해서 mode 그리고 loss 값을 training_step 그리고 validation_step 에서 만들어 줘야 합니다.
    checkpointer = ModelCheckpoint(
        dirpath='./checkpoints',
        filename='machine-translation-{mode}-{epoch:02d}-{step:06d}-{loss:.4f}',
        monitor='loss',
        every_n_epochs=1,
        save_top_k=-1,
        mode='min'
    )

    tensorboard = TensorBoardLogger('./tb_logs', name='machine-translation')

    trainer = Trainer(max_epochs=15,
                      accelerator='cuda',  # it's not GPU
                      devices=1,
                      log_every_n_steps=100,
                      enable_checkpointing=True,
                      enable_progress_bar=True,
                      enable_model_summary=True,
                      logger=tensorboard,  # TensorBoard 는 callback 에 넣는 것 아닙니다.
                      callbacks=[checkpointer])
    trainer.fit(model, train_loader, valid_loader)
    # trainer.validate(model, valid_loader)


if __name__ == '__main__':
    train()
