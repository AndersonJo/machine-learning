import logging

from lightning import Trainer
from torch.utils.data import DataLoader

from data import KorQuadDataset, InBatchNegativeSampler, KorquadCollator
from preprocess import _download_data, preprocess_raw_data
from model import KoBertBidirectionalEncoder

# 로거 생성 및 로그 레벨 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 콘솔 출력을 위한 핸들러 생성 및 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)


def train():
    data_paths = preprocess_raw_data()

    model = KoBertBidirectionalEncoder()
    train_dataset = KorQuadDataset(data_paths['train'])
    valid_dataset = KorQuadDataset(data_paths['valid'])
    test_dataset = KorQuadDataset(data_paths['test'])
    pad_id = train_dataset.pad_id

    train_loader = DataLoader(train_dataset,
                              batch_sampler=InBatchNegativeSampler(train_dataset, batch_size=32, drop_last=False),
                              collate_fn=KorquadCollator(pad_id=pad_id))
    valid_loader = DataLoader(valid_dataset, batch_size=32)

    trainer = Trainer(max_epochs=100,
                      reload_dataloaders_every_n_epochs=True)  # hard negative sampling 하려면 필요. LightningDataModule 의 train_dataloader 함수에서 Subset 사용해서 추가
    trainer.fit(model, train_loader, valid_loader)


if __name__ == '__main__':
    train()
