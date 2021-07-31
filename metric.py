import numpy as np
from sklearn import mixture
import pandas as pd
# from typing import NewType
from scCGAN.munkres import Munkres
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
def best_map(L1, L2):
    # L1 should be the groundtruth labels and L2 should be the clustering labels we got
    #unique 获取L1中的种类 -不重复的元素组成的数组
    Label1 = np.unique(L1)
    # print("L1", L1)
    # print("Label1",Label1)
    #L1的种类数目
    nClass1 = len(Label1)
    # print("nClass1", nClass1)
    Label2 = np.unique(L2)
    # print("L2", L2)
    # print("Label2", Label2)
    nClass2 = len(Label2)
    # print("nClass2", nClass2)
    nClass = np.maximum(nClass1, nClass2)
    # print("nClass", nClass)
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
        # print(c[i])
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
    ari_r = adjusted_rand_score(label_true_rare[0], label_pre_rare[0])
    nmi_r = normalized_mutual_info_score(label_true_rare[0], label_pre_rare[0])
    return acc, acc_r, ari_r, nmi_r
# def err_rate(gt_s, s):
#     c_x = best_map(gt_s, s)
#     # print("c_x",c_x)
#     # print(type(c_x))
#     err_x = np.sum(gt_s[:] != c_x[:])
#     missrate = err_x.astype(float) / (len(gt_s))
#     return missrate

# def compute_acc(y_true,y_pred):
#     missrate_x = err_rate(y_true,y_pred)
#     acc_x = 1 - missrate_x
#     return acc_x

def compute_purity(y_pred, y_true):
        """
        Calculate the purity, a measurement of quality for the clustering 
        results.
        
        Each cluster is assigned to the class which is most frequent in the 
        cluster.  Using these classes, the percent accuracy is then calculated.
        
        Returns:
          A number between 0 and 1.  Poor clusterings have a purity close to 0 
          while a perfect clustering has a purity of 1.

        """

        # get the set of unique cluster ids
        clusters = set(y_pred)

        # find out what class is most frequent in each cluster
        cluster_classes = {}
        correct = 0
        for cluster in clusters:
            # get the indices of rows in this cluster
            indices = np.where(y_pred == cluster)[0]

            cluster_labels = y_true[indices]
            majority_label = np.argmax(np.bincount(cluster_labels))
            correct += np.sum(cluster_labels == majority_label)
        
        return float(correct) / len(y_pred)

