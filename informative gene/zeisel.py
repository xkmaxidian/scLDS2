#z:39
import pandas as pd
import informative_genes
import numpy as np
from sklearn import preprocessing
import csv
#z:49
datapath="C:\\Users\\Administrator\\Desktop\\gene_zeisel\\Zeisel_tpm.csv"
data_all = pd.read_csv(datapath,index_col=False)
#删除含有nan的列
data = data_all.dropna(axis=1,how='all')
# print(data)

a=(data != 0).astype(int).sum(axis=0)
print(a)
b=a[a<20].index
data_s = data.drop(labels=b,axis=1)
print(data_s)
data = data_s

feature_path="C:\\Users\\Administrator\\Desktop\\gene_zeisel\\test_z.csv"
features = pd.read_csv(feature_path,header=None,index_col=False)
# print(features)

genes_feature = informative_genes.f(data,features)
genes_feature = genes_feature.T
# print(genes_feature.shape)

#reconstraction data
cell_features = features.as_matrix()
data_re = np.dot(cell_features,genes_feature)
print(data_re)
print(data_re.shape)


#normalize
min_max_scaler = preprocessing.MinMaxScaler()
datas= min_max_scaler.fit_transform(data_re)
columns = data.columns.values.tolist()
datas = pd.DataFrame(datas)
datas.columns = columns
print(datas)

#LABEL
label_path = 'C:\\Users\\Administrator\\Desktop\\gene_zeisel\\label_Predicted.txt'
labels = np.loadtxt(label_path).astype(int)
label=labels
print(label.shape)


#1.lasso feature extraction:
from sklearn.linear_model import Lasso
X_train = datas
y = label
names = data.columns.values.tolist()
lasso = Lasso(alpha=0.001)
lasso.fit(X_train, y)
print("Lasso model: ", informative_genes.pretty_print_linear(lasso.coef_, names, sort = True))
index =list( np.nonzero(lasso.coef_))
print(index)
feture_name = [names[i]for i in index[0]]
print(len(feture_name))
print(feture_name)
data_feature = data[feture_name]
print(data_feature)
data_feature.to_csv('zeisel_feature.csv', sep=',', header=True, index=True)



#2.lassoCV Feature extraction
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
X_train = datas
y = label
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 3))
    return(rmse)
model_lasso = LassoCV(alphas = [0.1,1,0.001, 0.0005]).fit(X_train, y)
print(model_lasso.alpha_)
print(model_lasso.coef_)
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
print(rmse_cv(model_lasso).mean())
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()