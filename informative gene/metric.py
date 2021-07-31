import numpy as np
from sklearn import mixture
import pandas as pd
from utils.munkres import Munkres
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import math
def best_map(L1, L2):
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
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
def err_rate_r(gt_s, s):
    c_x = best_map(gt_s, s)
    label_true_rare=[[]]
    label_pre_rare = [[]]
    rare = [1, 9, 10, 12, 13]
    for i in range(len(rare)):
        Y = np.where(gt_s == rare[i])
        print("Y",Y)
        labels_true = [gt_s[index] for index in Y]
        label_true_rare = np.hstack((label_true_rare , labels_true))
        labels_pre = [c_x[index] for index in Y]
        label_pre_rare = np.hstack((label_pre_rare, labels_pre))
    err_xr = np.sum(label_true_rare[0][:] != label_pre_rare[0][:])
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate_r = err_xr .astype(float) / (len(label_true_rare[0]))
    missrate = err_x.astype(float) / (len(gt_s))
    acc_r = 1 - missrate_r
    acc = 1-missrate
    ari_r = adjusted_rand_score(label_true_rare[0], label_pre_rare[0])
    nmi_r = normalized_mutual_info_score(label_true_rare[0], label_pre_rare[0])
    return acc, acc_r, ari_r, nmi_r

def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (len(gt_s))
    return missrate

def compute_acc(y_true,y_pred):
    missrate_x = err_rate(y_true,y_pred)
    acc_x = 1 - missrate_x
    return acc_x

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
        clusters = set(y_pred)
        cluster_classes = {}
        correct = 0
        for cluster in clusters:
            # get the indices of rows in this cluster
            indices = np.where(y_pred == cluster)[0]

            cluster_labels = y_true[indices]
            majority_label = np.argmax(np.bincount(cluster_labels))
            correct += np.sum(cluster_labels == majority_label)

        return float(correct) / len(y_pred)
def GaussianMixture_clustering(X_reduction):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 8)
    for n_components in n_components_range:
        gmm = mixture.GaussianMixture(n_components=n_components)
        gmm.fit(X_reduction)
        bic.append(gmm.bic(X_reduction))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
    bic = np.array(bic)
    clustering = best_gmm
    pred_label = clustering.predict(X_reduction)
    return pred_label

def data_reader():
    data_path = "C:\\Users\\Administrator\\ClusterGAN_Sc_data\\zeisel data\\zeisel_90%_tpm.csv"
    data = pd.read_csv(data_path)
    f = open(r'C:\\Users\\Administrator\\ClusterGAN_Sc_data\\zeisel data\\label_z.txt')
    n_cluster =7
    return data , f ,n_cluster



