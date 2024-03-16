import numpy as np
import pandas as pd


def data_preprocessing():
    data_paths = ['/home/anderson/Downloads/12.한영말뭉치/1_구어체(1)_200226.xlsx',
                  '/home/anderson/Downloads/12.한영말뭉치/1_구어체(2)_200226.xlsx']

    data = []
    for data_path in data_paths:
        df = pd.read_excel(data_path)
        df.set_index('SID', inplace=True)
        df.columns = ['korean', 'english']
        data.append(df)
    data = pd.concat(data)

    train_data, valid_data, test_data = np.split(
        data.sample(frac=1, random_state=42),
        [int(0.7 * len(data)), int(0.8 * len(data))])

    train_data.to_parquet('./_data/train.parquet')
    valid_data.to_parquet('./_data/valid.parquet')
    test_data.to_parquet('./_data/test.parquet')


def create_sentencepiece_data():
    df = pd.read_parquet('./_data/data.parquet')
    with open('./_data/korean.txt', 'wt') as f:
        for i, row in df.iterrows():
            f.write(row.korean)
            f.write('\n')

    with open('./_data/english.txt', 'wt') as f:
        for i, row in df.iterrows():
            f.write(row.english)
            f.write('\n')


if __name__ == '__main__':
    data_preprocessing()
    # create_sentencepiece_data()
