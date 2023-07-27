import os
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import torchvision
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from itertools import product
import math
import copy
import time
import logging
import pickle
import random
import sys

from datasets import TS_truncated
#, MNIST_truncated, EMNIST_truncated, CIFAR10_truncated, CIFAR10_Poisoned, CIFAR10NormalCase_truncated, EMNIST_NormalCase_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #output = F.log_softmax(x, dim=1)
        return x
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        # 可以选择"w"
        self.log = open(filename, "a", encoding="utf-8")  # 防止编码错误

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def reset(self):
        self.log.close()
        sys.stdout = self.terminal

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def record_net_data_stats(y_train, net_dataidx_map):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


def load_UCR_data(datadir, dataset):
    train_data = np.loadtxt(datadir + dataset +'/'+dataset + '_TRAIN.txt')
    test_data = np.loadtxt(datadir + dataset + '/' + dataset + '_TEST.txt')

    X_train, y_train = train_data[:, 1:],train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]
    print('load UCR data', X_train.shape, X_test.shape)
    return (X_train, y_train, X_test, y_test)

def partition_tsdata(dataset, datadir, partition, n_nets, alpha):
    # partition_strategy = "homo"
    # partition_strategy = "hetero-dir"
    print('---------------load UCR daset-------------------')
    X_train, y_train, X_test, y_test = load_UCR_data(datadir, dataset)
    n_train = X_train.shape[0]
    if partition == "homo":
        idxs = np.random.permutation(n_train) # 随机排序
        batch_idxs = np.array_split(idxs, n_nets) # 把idxs分为 nnets份
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)} #用户id->数据的字典
    elif partition == "hetero-dir":
        min_size = 0
        K = 10
        K = len(np.unique(y_train))
        N = y_train.shape[0]
        net_dataidx_map = {}
        while (min_size < 1) or (dataset == 'mnist' and min_size < 100):
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0] # 取出10个类分别对应的下标集合
                np.random.shuffle(idx_k) # 打乱下标，重复alphanets次
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets)) # 地雷克雷分布
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum() # 归一化
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # 数据采用地雷克雷分布分配给用户
        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    # fanhui
    return net_dataidx_map


def get_ts_loader(dataset, datadir, train_bs, test_bs, dataidxs=None):

    dl_obj = TS_truncated

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True)
    test_ds = dl_obj(datadir, train=False)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    # print('get ts loader length: ', len(train_ds), len(test_ds))

    return train_dl, test_dl


from torch.utils.data import TensorDataset, DataLoader

def load_ucr(path, normalize=False):
    data = np.loadtxt(path)
    # data[:, 0] -= 1
    # limit label to [0,num_classes-1]
    num_classes = len(np.unique(data[:, 0]))
    for i in range(data.shape[0]):
        if data[i, 0] < 0:# 标签小于0则重置为num_classes - 1
            data[i, 0] = num_classes - 1
    # Normalize some datasets without normalization在没有归一化的情况下使某些数据集归一化
    if normalize:
        mean = data[:, 1:].mean(axis=1, keepdims=True)
        std = data[:, 1:].std(axis=1, keepdims=True)
        data[:, 1:] = (data[:, 1:] - mean) / (std + 1e-8)
    return data # 返回归一化数据
from torch.utils.data import Dataset
class UcrDataset(Dataset):
    def __init__(self, txt_file, channel_last, normalize):
        '''
        :param txt_file: path of file
        :param channel_last
        '''
        # self.data = np.loadtxt(txt_file)
        self.data = load_ucr(txt_file, normalize)
        self.channel_last = channel_last
        if self.channel_last:
            self.data = np.reshape(self.data, [self.data.shape[0], self.data.shape[1], 1])
        else:
            self.data = np.reshape(self.data, [self.data.shape[0], 1, self.data.shape[1]])

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        if not self.channel_last:
            return self.data[idx, :, 1:], self.data[idx, :, 0]
        else:
            return self.data[idx, 1:, :], self.data[idx, 0, :]

    def get_seq_len(self):
        if self.channel_last:
            return self.data.shape[1] - 1
        else:
            return self.data.shape[2] - 1

def seed_experiment(seed=0):
    # seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Seeded everything %d",seed)

def load_UEA(dataset):
    # 读取训练和测试数据
    print(f'data/UCR/{dataset}/{dataset}_TRAIN.arff')
    train_data = loadarff(f'data/UCR/{dataset}/{dataset}_TRAIN.arff')[0]
    # print('T')
    test_data = loadarff(f'data/UCR/{dataset}/{dataset}_TEST.arff')[0]
    # file = open("./class/" + dataset + "zengqiangclass.xls", 'w')
    # file.write(dataset + '\n')
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    # 获取对应类别的下标
    def findid(y, target):#类别：标签
        index = []
        for i, lab in enumerate(y):
            if lab == target:
                index.append(i)
        return index

    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    #print(train_y)
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    labels = np.unique(train_y) #去除重复元素
    '''indexes = []
    for label in labels:
        index = findid(train_y, label)
        # file.write(str(len(index)) + '\n')
        print(label, len(index))
        index = random.sample(index, int(0.5*len(index))) # 随机选取80%的数据
        indexes.extend(index)'''
    #print(indexes, len(indexes))
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    '''train_X = train_X[indexes]
    train_y = train_y[indexes]'''
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    test_y = np.vectorize(transform.get)(test_y)
    # file.close()
    return train_X, train_y, test_X, test_y