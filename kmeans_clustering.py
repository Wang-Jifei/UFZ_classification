import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
#from sklearn import


def loadDataSet(fileName):
    data = np.loadtxt(fileName,delimiter=' ')
    #data = np.loadtxt(fileName, delimiter='\t', dtype=float, skiprows=1)
    return data
    #data =  np.loadtxt(fileName)


input_dir=r'lda_result/doc_topic_distribution.txt'
dataset = loadDataSet(input_dir)
print(dataset)
print(dataset.shape)
X=dataset
#绘制数据分布图
plt.scatter(X[:, 0], X[:, 1], c = "red", marker='o', label='data')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()
X2=dataset[:,2:]#选取最后两列特征聚类
estimator = KMeans(n_clusters=6)#构造聚类器
estimator.fit(X)#聚类
label_pred = estimator.labels_ #获取聚类标签
result_txt = open("lda_result/kmeans_result.txt",'w+')
for i in label_pred:
    result_txt.write("{}\n".format(int(i)))

result_txt.close()

#绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
x3 = X[label_pred == 3]
x4 = X[label_pred == 4]
x5 = X[label_pred == 5]
plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='label2')
plt.scatter(x3[:, 0], x3[:, 1], c = "yellow", marker='^', label='label3')
plt.scatter(x4[:, 0], x4[:, 1], c = "maroon", marker='s', label='label4')
plt.scatter(x5[:, 0], x5[:, 1], c = "aqua", marker='X', label='label5')

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()