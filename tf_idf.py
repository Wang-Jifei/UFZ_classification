from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

"""
Calculate tf-idf
"""
# 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = CountVectorizer(max_features=100)
input_dir=r'Feature vector/word_frequency_file.txt'
output_dir= r'Feature vector/word_tfidf.txt'
data_mat = np.loadtxt(input_dir)

# 该类会统计每个词语的tf-idf权值
tf_idf_transformer = TfidfTransformer()
# 将文本转为词频矩阵并计算tf-idf
tf_idf = tf_idf_transformer.fit_transform(data_mat)
# 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
x_weight = tf_idf.toarray()


print('print vector：')
print(x_weight)
np.savetxt(output_dir, x_weight, fmt='%6f', delimiter=' ')
