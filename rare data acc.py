import numpy as np
from sklearn import mixture
import pandas as pd
# from typing import NewType
from ClusterGANmaster.munkres import Munkres

def best_map(L1, L2):
    # L1 should be the groundtruth labels and L2 should be the clustering labels we got
    #unique 获取L1中的种类 -不重复的元素组成的数组
    Label1 = np.unique(L1)
    #L1的种类数目
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    #生成nClass*nClass大小的矩阵
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        #取出L1中等于Label1[i]的索引（属于某一类的索引）
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    #gt_s 是真实label，c_x 是匹配之后的预测label，如果 我们仅仅想计算稀有的话 在这里取出相应的所有的稀有label，并且组成一个新的数组，计算acc
    label_true_rare=[[]]
    label_pre_rare = [[]]
    rare=[7,9,11,12,13,14]
    for i in range(len(rare)):
        Y = np.where(gt_s == rare[i])
        labels_true = [gt_s[index] for index in Y]
        label_true_rare = np.hstack((label_true_rare , labels_true))
        labels_pre = [c_x[index] for index in Y]
        label_pre_rare = np.hstack((label_pre_rare, labels_pre))
    print("len(label_pre_rare)",len(gt_s))
    print("len(label_true_rare)",len(c_x))
    print("len(label_pre_rare)",len(label_pre_rare[0]))
    print("len(label_true_rare)",len(label_true_rare[0]))
    err_xr = np.sum(label_true_rare[0][:] != label_pre_rare[0][:])
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate_r = err_xr .astype(float) / (len(label_true_rare[0]))
    missrate = err_x.astype(float) / (len(gt_s))
    acc_r = 1 - missrate_r
    acc = 1-missrate
    return acc, acc_r

if __name__=='__main__':
    label_true_path= 'C:\\Users\\Administrator\\label.txt'
    label_true = np.loadtxt(label_true_path).astype(int)
    # labels_t=[[]]
    # Y = np.where(label_true==1)
    # labels_t1 = [label_true[index] for index in Y]
    # labels_t = np.hstack((labels_t, labels_t1))
    # Y = np.where(label_true == 2)
    # labels_t2 = [label_true[index] for index in Y]
    # labels_t=np.hstack(( labels_t, labels_t2))
    # print( len(labels_t[0]))
    # print(labels_t[0])
    # print(len(labels[0]))
    # print(len(Y[0]))
    label_pre_path="E:\\code data\\TSNE_DATA\\all\\label_Predicted.txt"
    label_pre = np.loadtxt(label_pre_path).astype(int)
    print(len(label_true))
    print(len(label_pre))
    acc , acc_r  = err_rate(label_true,label_pre)
    print("acc",acc)
    print("acc_r",acc_r)