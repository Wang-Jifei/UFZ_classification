###cluster.py
# 导入相应的包

import scipy.cluster.hierarchy as sch

import numpy as np
import matplotlib.pylab as plt
def loadDataSet(fileName):
    data = np.loadtxt(fileName,delimiter=' ')
    #data = np.loadtxt(fileName, delimiter='\t', dtype=float, skiprows=1)
    return data
    #data =  np.loadtxt(fileName)

# 读取数据
input_dir=r'lda_result/doc_topic_distribution_500_1k.txt'
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
