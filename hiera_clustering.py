import math
import numpy as np
import sklearn
from sklearn.datasets import load_iris
import os
def loadDataSet(fileName):
    data = np.loadtxt(fileName,delimiter=' ')
    #data = np.loadtxt(fileName, delimiter='\t', dtype=float, skiprows=1)
    return data
    #data =  np.loadtxt(fileName)
from sklearn.cluster import AgglomerativeClustering
"""
if __name__ == '__main__':
    data = [[16.9,0],[38.5,0],[39.5,0],[80.8,0],[82,0],[834.6,0],[116.1,0]]
    input_dir=r'lda_result/doc_topic_distribution.txt'
    dataset = loadDataSet(input_dir)

    clustering = AgglomerativeClustering(n_clusters=7).fit(dataset)
    print(clustering.labels_)
    print(clustering.children_)
"""
###cluster.py
# 导入相应的包
import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq, kmeans, whiten
import numpy as np
import matplotlib.pylab as plt

# 生成待聚类的数据点,这里生成了20个点,每个点4维:
data = [[16.9, 0], [38.5, 0], [39.5, 0], [80.8, 0], [82, 0], [834.6, 0], [116.1, 0]]
input_dir=r'lda_result/doc_topic_distribution.txt'
dataset = loadDataSet(input_dir)
# 加一个标签进行区分
A = []
for i in range(len(dataset)):
    a = chr(i+ord('A'))
    A.append(a)

Z = sch.linkage(dataset, 'ward')
f = sch.fcluster(Z, t=0.5, criterion='distance')  # 聚类，这里t阈值的选择很重要
print(f)  #打印类标签
fig = plt.figure(figsize=(10,6))
dn = sch.dendrogram(Z,labels=A)
plt.show()
result_txt = open("lda_result/Hcluster_result.txt",'w+')
for i in f:
    result_txt.write("{}\n".format(int(i)))

result_txt.close()
"""
# 将层级聚类结果以树状图表示出来并保存为plot_dendrogram.png
fig = plt.figure()
P = sch.dendrogram(Z, labels=A)
plt.show()

print(Z)
"""
