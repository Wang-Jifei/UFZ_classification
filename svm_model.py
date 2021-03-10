# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from time import time   #对程序运行时间计时用的
import logging           #打印程序进展日志用的
import matplotlib.pyplot as plt  #绘图用的

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np

# load dataset
def loadDataSet(filename):
    fr = open(filename,'rb')
    data = []
    label = []
    mat_txt= np.loadtxt(filename)
    mat_data= mat_txt[:,:-1]
    mat_label=mat_txt[:,-1]
    for line in fr.readlines():
        lineAttr = line.decode().strip().split('\t')
        data.append([float(x) for x in lineAttr[:-1]])
        label.append(float(lineAttr[-1]))
    return mat_data,mat_label

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')

# define data and label
data, label = loadDataSet('lda_result/label_result.txt') # add class types as the last column in the doc_topic distribution matrix
X = data   # 特征向量矩阵
y = label
n_feature=X.shape[1]#每个人有多少个特征值
n_samples=X.shape[0]#总共有多少个样本

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2, test_size=0.2)

# training a SVM classifier
# set SVM training parameters
param_grid ={'C':[1e4,5e2,1e2,5e2,1e5], # C是对错误的惩罚[1e3,5e3,1e4,5e4,1e5]
             'gamma':[0.01,0.001,0.005,0.05,0.1,0.1],} # gamma核函数里多少个特征点会被使用}#对参数尝试不同的值[0.0001,0.0005,0.001,0.005,0.001,0.001]

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

print(cm) #print confusion matrix