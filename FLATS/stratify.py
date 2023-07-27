# -*- coding:utf-8 -*-
import numpy as np
import os
from collections import Counter
import random

import matplotlib.pyplot as plt
from pyts.transformation import ShapeletTransform
from ts_attack.query_probalility import load_ucr


def shapelet_transform(run_tag):
    # Shapelet transformation
    st = ShapeletTransform(n_shapelets=5, window_sizes=[0.1, 0.2, 0.3, 0.4
                                                        ], sort=True, verbose=1, n_jobs=1)
    path = './data/UCR/' + run_tag + '/' + run_tag + '_attack.txt'
    data = load_ucr(path)
    X = data[:, 1:]
    mask = np.isnan(X)
    X[mask] = 0
    y = data[:, 0]
    print('shapelet transfor')
    X_new = st.fit_transform(X, y)
    print('save shapelet pos')
    file = open('./shapelet_pos/' + run_tag + '_shapelet_pos.txt', 'w+')
    for i, index in enumerate(st.indices_):
        idx, start, end = index
        file.write(run_tag + ' ')
        file.write(str(idx) + ' ')
        file.write(str(start) + ' ')
        file.write(str(end) + '\n')
    file.close()
    # Visualize the most discriminative shapelets
    plt.figure(figsize=(6, 4))
    for i, index in enumerate(st.indices_):
        idx, start, end = index
        plt.plot(X[idx], color='C{}'.format(i),
                 label='Sample {}'.format(idx))
        plt.plot(np.arange(start, end), X[idx, start:end],
                 lw=5, color='C{}'.format(i))

    plt.xlabel('Time', fontsize=12)
    plt.title('The five more discriminative shapelets', fontsize=14)
    plt.legend(loc='best', fontsize=8)
    plt.savefig('./shapelet_fig/' + run_tag + '_shapelet.pdf')
    # plt.show()

def stratify_datasets(dataname):
    #
    data_path = './data/UCR/' + dataname + '/' + dataname
    train_data = np.loadtxt(data_path + '_TRAIN.txt')
    test_data = np.loadtxt(data_path + '_TEST.txt')
    data = np.vstack((train_data, test_data))
    print('data: ', test_data.shape, train_data.shape, data.shape)
    test_count = Counter(data[:, 0])
    size = data.shape[0]

    def findid(y, target):  # 类别：标签
        index = []
        for i, lab in enumerate(y):
            if lab == target:
                index.append(i)
        return index
    indexes = []
    for label in test_count.keys():
        index = findid(data[:, 0], label)
        # file.write(str(len(index)) + '\n')
        # print(label, len(index))
        index = random.sample(index, int(0.400*len(index))) # 随机选取40%的数据测试
        indexes.extend(index)
    if 0 not in test_count.keys():
        print("label -1 ;limit label to [0,num_classes-1]")
        data[:, 0] -= 1
        # limit label to [0,num_classes-1]
        num_classes = len(np.unique(data[:, 0]))
        for i in range(data.shape[0]):
            if data[i, 0] < 0:  # 标签小于0则重置为num_classes - 1
                data[i, 0] = num_classes - 1
    eval_data = data[indexes]
    no_indexes = np.delete(np.arange(size), indexes)
    print(type(indexes))
    print(type(no_indexes))
    traindataset = data[no_indexes]

    np.savetxt(data_path+ '_TEST.txt', eval_data)
    np.savetxt(data_path + '_TRAIN.txt', traindataset)
    print(type(eval_data))

    print('testdata: ', data.shape)

    print('test class num: ', test_count)
    train_data = np.loadtxt(data_path + '_TRAIN.txt')
    test_data = np.loadtxt(data_path + '_TEST.txt')
    print('data after:test,train ', test_data.shape, train_data.shape)


def stratify_attack_data(dataname):
    data_path = './data/UCR/' + dataname + '/' + dataname
    data = np.loadtxt(data_path + '_TRAIN.txt')
    size = data.shape[0]
    test_count = Counter(data[:, 0])
    print(test_count)
    def findid(y, target):
        index = []
        for i, lab in enumerate(y):
            if lab == target:
                index.append(i)
        return index
    label = 0
    index = findid(data[:, 0], label)
    num_att = int(0.05 * data.shape[0])
    ind_att = random.sample(index, num_att)  #
    attack_data = data[ind_att]
    no_attack = np.delete(np.arange(size), ind_att)
    print(type(ind_att))
    print(type(no_attack))
    eval_data = data[no_attack]

    np.savetxt(data_path+ '_no_attack.txt', eval_data)
    np.savetxt(data_path + '_attack.txt', attack_data)
    print(type(eval_data))

    # eval_data = np.loadtxt(data_path + '_eval.txt')
    # unseen_data = np.loadtxt(data_path + '_unseen.txt')

    print('testdata: ', data.shape)

    print('train: attack data: ', eval_data.shape, attack_data.shape)

if __name__ == '__main__':

    names = ['ECG5000']

    for name in names:
        stratify_datasets(name)
        #stratify_attack_data(name)
        #shapelet_transform(name)