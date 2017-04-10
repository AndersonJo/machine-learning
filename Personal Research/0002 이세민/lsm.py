# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

file_name = '/dataset/personal_projects/lsm/20160123_1차 발주_3280.xlsx'
N_DAS = 96
SHEET_NAME = 'original'
FINDING_LOCATION_TIME = 9
FINDING_PRODUCT_TIME = 12


class LSM(object):
    EMPTY_LOCATION = 'Z1-1'

    def __init__(self, n_das=N_DAS, sheet=SHEET_NAME):
        self.data = None
        self.vectors = None
        self.N_DAS = n_das
        self.sheet = SHEET_NAME

        self.barcode_to_index = None
        self.index_to_barcode = None

        self.location_to_idx = None
        self.idx_to_location = None

        self.barcode_vectors = None
        self.location_vectors = None

    def load(self, file_name):
        COL_NAMES = ['date', 'order_number', 'brand_code', 'brand_name',
                     'product', 'property', 'location', 'barcode', 'n', 'price',
                     'status', 'orderer', 'recipient', 'address', 'postcode',
                     'contact1', 'contact2', 'comment', 'part_delivery', 'code']
        self.data = pd.read_excel(file_name,
                                  names=COL_NAMES,
                                  sheetname=self.sheet)


    # def frequent_barcodes(self):
    #     frq_barcodes = self.data[['barcode', 'n']].groupby(by='barcode').sum()
    #     frq_barcodes = frq_barcodes.sort_values(by='n', ascending=False)
    #     codes = []
    #     for barcode in frq_barcodes.index:
    #         for code in self.data[self.data['barcode'] == barcode]['code'].values:
    #             if code not in codes:
    #                 codes.append(code)
    #     return codes

    def remove_useless_barcodes(self):
        # Barcode가 1개인 것을 찾음
        # 즉 물건을 찾으러 가서 바로 DAS이용할 필요 없이 바로 물품 보낼수 있는 것
        n_order = self.barcode_vectors.shape[0]
        useless_indices = []
        useless_codes = []
        for i in range(n_order):
            r = self.barcode_vectors * self.barcode_vectors[i]
            r = np.sum(r, axis=1)
            r = (r > 0).sum()
            if r <= 1:
                useless_indices.append(i)
                useless_codes.append(self.index_to_code[i])

        useless_indices = np.array(useless_indices)
        useful_indices = np.array([i for i in range(n_order) if i not in useless_indices])
        self.barcode_vectors = self.barcode_vectors[useful_indices]






    def frequent_code_index(self):
        # 가장 빈번한 barcode를 찾음
        summed = np.sum(self.barcode_vectors, axis=0)
        most_frequent_barcode_idx = np.argmax(summed)
        most_frequent_code_idx = np.argmax(self.barcode_vectors[:, most_frequent_barcode_idx])
        return most_frequent_code_idx

    def preprocess(self):
        # Pre-Process missing locations
        self.data['location'] = self.data['location'].fillna(self.EMPTY_LOCATION)

        # Split Locations to l1, l2, l3
        location_regex = '(?P<l1>[a-zA-Z]+)-?(?P<l2>\d+)-(?P<l3>\d+)'
        self.data['location'][~self.data['location'].str.contains(location_regex)][:] = self.EMPTY_LOCATION
        location_series = self.data['location'].str.extract(location_regex, expand=False)
        self.data = pd.concat([self.data, location_series], axis=1)

        # Create Barcode Vectors
        n_uniq_code = len(self.data['code'].unique())
        n_uniq_barcode = len(self.data['barcode'].unique())
        barcode_vectors = np.zeros((n_uniq_code, n_uniq_barcode))  # (N 주문, N 바코드 즉 상품)

        self.code_to_index = {code: i for i, code in enumerate(self.data['code'].unique())}
        self.index_to_code = {idx: code for code, idx in self.code_to_index.items()}

        self.barcode_to_index = {barcode: i for i, barcode in enumerate(self.data['barcode'].unique())}
        self.index_to_barcode = {idx: barcode for barcode, idx in self.barcode_to_index.items()}

        for barcode, code, n in self.data[['barcode', 'code', 'n']].values:
            code_idx = self.code_to_index[code]
            barcode_idx = self.barcode_to_index[barcode]
            barcode_vectors[code_idx, barcode_idx] += n

        # Create Location Vectors
        unique_locations = list()
        for l1, l2, l3 in self.data[['l1', 'l2', 'l3']].values:
            l1 = 'l1_' + str(l1)
            l2 = 'l2_' + str(l2)
            l3 = 'l3_' + str(l3)
            if l1 not in unique_locations:
                unique_locations.append(l1)

            if l2 not in unique_locations:
                unique_locations.append(l2)

            if l3 not in unique_locations:
                unique_locations.append(l3)

        n_unique_location = len(unique_locations)
        self.location_to_idx = {loc: i for i, loc in enumerate(unique_locations)}
        self.idx_to_location = {i: loc for loc, i in self.location_to_idx.items()}
        location_vectors = np.zeros((n_uniq_code, n_unique_location))  # 주문별 위치의 vector 값 (N 주문, N 위치)

        for code, l1, l2, l3 in self.data[['code', 'l1', 'l2', 'l3']].values:
            code_idx = self.code_to_index[code]
            l1_idx = self.location_to_idx['l1_' + str(l1)]
            l2_idx = self.location_to_idx['l2_' + str(l2)]
            l3_idx = self.location_to_idx['l3_' + str(l3)]
            location_vectors[code_idx, l1_idx] += 1
            location_vectors[code_idx, l2_idx] += 1
            location_vectors[code_idx, l3_idx] += 1

        # Normalization
        self.barcode_scaler = MinMaxScaler()
        self.location_scaler = MinMaxScaler()

        self.barcode_vectors = self.barcode_scaler.fit_transform(barcode_vectors)
        self.location_vectors = self.barcode_scaler.fit_transform(location_vectors)

    def get_code(self, code):
        return self.data[self.data['code'] == code]

    def process(self):
        # Init
        code_idx = self.frequent_code_index()
        n_barcode = self.barcode_vectors.shape[0]

        # Init for Barcode Vectors
        barcode_vectors = self.barcode_vectors.copy()
        barcode_acum = np.zeros(self.barcode_vectors.shape[-1])
        barcodes_indices = [code_idx]

        # Search Similar Barcodes
        for i in range(n_barcode):
            bvector = self.barcode_vectors[code_idx]
            barcode_vectors[code_idx] = 0

            n_barcode = np.sum(bvector > 0)
            barcode_acum += bvector / n_barcode

            sim_scores = self.sim(barcode_vectors, barcode_acum)
            code_idx = np.argmax(sim_scores)
            code = self.index_to_code[code_idx]
            score = sim_scores[code_idx]
            barcodes_indices.append(code_idx)

            print(i, score, code_idx, np.sum(barcode_acum))

    def sim(self, a, b):
        # Sigmoid
        return 1. / (1 + np.e ** np.sum(-np.multiply(a, b), axis=1)) - 0.5

    def _get_barcode_vector(self, code_idx, g):
        return self.barcode_vectors[code_idx]


lsm = LSM()
lsm.load(file_name)
lsm.preprocess()
# frq_codes = lsm.frequent_barcodes()
# start_code = frq_codes[0] # 2849240

lsm.remove_useless_barcodes()
