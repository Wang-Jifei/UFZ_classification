
from time import time    # 对程序运行时间计时
import logging           # 打印程序进展
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
""" ######### 1. load dataset ######### """

def loadDataSet(datatxt,labeltxt):
    f1 = open(datatxt,'rb')
    f2 = open(labeltxt, 'rb')
    data = []
    label = []
    np_data = np.loadtxt(datatxt)
    np_label = np.loadtxt(labeltxt)
    mat_data = np_data
    mat_label = np_label
    for line in f1.readlines():
        lineAttr = line.decode().strip().split(' ')
        data.append([float(x) for x in lineAttr])
    for line in f2.readlines():
        lineAttr = line.decode().strip().split(' ')
        label.append(float(lineAttr[0]))
    return mat_data, mat_label

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')

""" ######### 2. define data and label ######### """
data_dir = r'lda_result/doc_topic_distribution.txt'
label_dir = r'data/ufz_class.txt'
data, label = loadDataSet(data_dir,label_dir)
X = data   # 特征向量矩阵
y = label
n_feature=X.shape[1]# 有多少个特征值
n_samples=X.shape[0]# 总共有多少个样本


# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,test_size=0.2)

# training a GBDT classifier
gbdt = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=500, subsample=1
                                  , min_samples_split=2, min_samples_leaf=8, max_depth=16
                                  , init=None, random_state=None, max_features=None
                                  , verbose=0, max_leaf_nodes=None, warm_start=False)
gbdt.fit(X_train, y_train)
Y_pred = gbdt.predict(X_test)
# model accuracy for X_test
accuracy = gbdt.score(X_test, y_test)

print(accuracy)
# creating a confus.iix(y_test, Y_pred)
cm=confusion_matrix(y_test,Y_pred)

print(cm)
# Write the result
accuracy_f=open("lda_result/gbdt_result.txt",'w+')
print(cm)
accuracy_f.write("{}\n".format(accuracy))
for i in range(len(cm[0])):
    for j in range(len(cm)):
        accuracy_f.write("{}\t".format(cm[i][j]))
    accuracy_f.write("\n")
accuracy_f.close()

# Draw confusion matrix

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
sns.set()
f,ax=plt.subplots()
sns.heatmap(cm,annot=True,ax=ax,cmap='Blues') #画热力图

ax.set_title('confusion matrix') #标题
ax.set_xlabel('predict') #x轴
ax.set_ylabel('true') #y轴
plt.show()
