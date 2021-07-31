import scipy.sparse
import scipy.io
import numpy as np
import sys
import pandas as pd
import math

class DataSampler(object):
    def __init__(self):
        self.total_size = 3603
        self.train_size = 0
        self.test_size = 652
        self.X_train, self.X_test = self._load_gene_data()
        self.y_train, self.y_test = self._load_labels()


    def _read_mtx(self, filename):
        buf = scipy.io.mmread(filename)
        return buf
    def _load_gene_data(self):
        data_path = 'E:\\7.31z\\7.31\\scCGAN\\scCGAN\\scCGAN\\data\\Braon\\GSM2230759_human3.csv'
        data = pd.read_csv(data_path, header=-1)
        data = np.array(data)
        idx = np.flatnonzero(data)
        N = math.floor(0.001 * len(idx))
        print(data)
        # print(N)
        np.put(data, np.random.choice(idx, size=N, replace=False), 0)
        print(data.shape)
        data_train = data[0:self.train_size, :]
        data_test = data[self.train_size:, :]
        return data_train, data_test

    def _load_labels(self):
        data_path = 'E:\\7.31z\\7.31\\scCGAN\\scCGAN\\scCGAN\\data\\Braon\\label.txt'
        labels = np.loadtxt(data_path).astype(int)
        print(labels.shape)

        # np.random.seed(0)
        # indx = np.random.permutation(np.arange(self.total_size))
        labels_train = labels[0:self.train_size]
        labels_test = labels[self.train_size:]
        return labels_train, labels_test

    def train(self, batch_size, label=False):
        # indx = np.random.randint(low = 0, high = self.train_size, size = batch_size)

        if label:
            return self.X_train[self.train_size:, :], self.y_train[self.train_size:, :].flatten()
        else:
            return self.X_train[self.train_size:, :]

    def validation(self):
        return self.X_train[-250:, :], self.y_train[-250:].flatten()

    def test(self):
        return self.X_test, self.y_test

    def load_all(self):
        return np.concatenate((self.X_train, self.X_test)), np.concatenate((self.y_train, self.y_test))


#     def _load_gene_mtx(self):
#         ''' The entries of the counts matrix C are first tranformed as log2(1 + Cij )and then divided by the maximum entry of the transformation to obtain values in the range of [0, 1].'''
#
#         data_path = 'C:/Users/Administrator/ClusterGAN/ClusterGAN-master/ClusterGAN-master/10x_73k/10x_73k/sub_set-720.mtx'
#         data = self._read_mtx(data_path)
#         data = data.toarray()
#         print(data)
#         data = np.log2(data + 1)
#         scale = np.max(data)
#         data = data / scale
# #np.random.permutation()：随机排列序列。打乱数据，分组train和test.
#         np.random.seed(0)
#         indx = np.random.permutation(np.arange(self.total_size))
#         data_trains = data[indx[0:self.train_size], :]
#         data_tests = data[indx[self.train_size:], :]
#
#         return data_trains, data_tests
#     def _load_gene_data(self):
#         data_path='E:\\code data\\ClusterGAN_Sc_data_zan\\human3_shaixuan10\\GSM2230759_human3_10_tpm.csv'
#         data = pd.read_csv(data_path)  # 读取csv文件
#         data = np.array(data)
#         print(data)
#         print(data.shape)
#         # data = np.log2(data + 1)
#         # scale = np.max(data)
#         # data = data / scale
#         # np.random.permutation()：随机排列序列。打乱数据，分组train和test.
#         np.random.seed(0)
#         indx = np.random.permutation \
#             (np.arange(self.total_size))
#         data_train = data[indx[0:self.train_size], :]
#         data_test = data[indx[self.train_size:], :]
#         print(data_train)
#         return data_train, data_test
#
#     def _load_labels(self):
#         data_path = 'C:\\Users\\Administrator\\label.txt'
#         labels = np.loadtxt(data_path).astype(int)
#         print(labels.shape)
#         np.random.seed(0)
#         indx = np.random.permutation(np.arange(self.total_size))
#         labels_train = labels[indx[0:self.train_size]]
#         labels_test = labels[indx[self.train_size:]]
#         return labels_train, labels_test

       
    # def train(self, batch_size, label = False):
    #     # indx = np.random.randint(low = 0, high = self.train_size, size = batch_size)
    #
    #     if label:
    #         return self.X_train[indx, :], self.y_train[indx].flatten()
    #     else:
    #         return self.X_train[indx, :]
    #
    # def validation(self):
    #     return self.X_train[-250:,:], self.y_train[-250:].flatten()
    #
    # def test(self):
    #     return self.X_test, self.y_test
    #
    # def load_all(self):
    #      return np.concatenate((self.X_train, self.X_test)), np.concatenate((self.y_train, self.y_test))

if __name__=='__main__':

    data = DataSampler()
    # print(data.X_train[0])
    # print(len(data.X_train[0]))
    # print(data.y_train)
    # print(len(data.y_train))
    # print(data.y_train[0])
    # data_path = 'C:/Users/Administrator/ClusterGAN/ClusterGAN-master/ClusterGAN-master/10x_73k/10x_73k/sub_set-720.mtx'
    # buf = data._read_mtx(data_path)
    # print(buf[0])
