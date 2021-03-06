# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from time import time   #对程序运行时间计时
import logging           #打印程序进展日志
import matplotlib.pyplot as plt  #绘图

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np

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
data, label = loadDataSet(data_dir,label_dir) # add class types as the last column in the doc_topic distribution matrix
X = data   # 特征向量矩阵
y = label
n_feature=X.shape[1]# 每个人有多少个特征值
n_samples=X.shape[0]# 总共有多少个样本

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# training a SVM classifier
# set SVM training parameters
param_grid ={'C':[1e3,5e3,1e4,5e4,1e5], # C是对错误的惩罚[1e3,5e3,1e4,5e4,1e5]
             'gamma':[0.0001,0.0005,0.001,0.005,0.001,0.001],} # gamma核函数里多少个特征点会被使用
                                                       # 对参数尝试不同的值[0.0001,0.0005,0.001,0.005,0.001,0.001]

svm_model = GridSearchCV(SVC(kernel='rbf'),param_grid)
svm_train = svm_model.fit(X_train,y_train)
svm_predictions = svm_train.predict(X_test)
# svm_train = SVC(kernel='linear', C=1).fit(X_train, y_train)
# svm_predictions = svm_train.predict(X_test)

# model accuracy for X_test
accuracy = svm_train.score(X_test, y_test)
print(accuracy)
# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)

print(cm) # print confusion matrix

# Write the result
accuracy_f=open("lda_result/svm_result.txt",'w+')
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