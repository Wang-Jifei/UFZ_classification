import os
import cv2
import time
import numpy as np
from sklearn import svm

import joblib
from sklearn.decomposition import LatentDirichletAllocation # LDA model

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import operator
import glob    # readfile
import shutil  # copyfile

font = FontProperties(fname=r"C:\Windows\Fonts\Calibri.ttf", size=14)

"""
Step1. Calculate the SIFT feature from raw block image

Return: SIFT features
"""
def calcSiftFeature(img):
    # rgb2grey
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SURF_create()
    # calculate feature points and description
    keypoints, features = sift.detectAndCompute(gray, None)
    return features


"""
Step2. Calculate Kmeans features from SIFT features of ALL images

Return: K centers
"""

def learnVocabulary(features,K):
    # criteria表示迭代停止的模式   eps---精度0.1，max_iter---满足超过最大迭代次数20
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    # k-means random centers
    flags = cv2.KMEANS_RANDOM_CENTERS
    # 标签，中心 = kmeans(输入数据（特征)、聚类的个数K,预设标签，聚类停止条件、重复聚类次数、初始聚类中心点
    #print('feature_shape',features.shape)
    compactness, labels, centers = cv2.kmeans(features,K, None,criteria, 20, flags)
    #print ("generated vocabulary Done")
    return centers

"""
Step2. Find corresponding centers of each image

Return: feature vector
"""


# calculate feature vector from Bow
def calcFeatVec(features, centers,K):
    featVec = np.zeros((1, K))
    for i in range(0, features.shape[0]):
        #第i张图片的特征点
        fi = features[i]
        diffMat = np.tile(fi, (K, 1)) - centers
        #axis=1按行求和，即求特征到每个中心点的距离
        sqSum = (diffMat**2).sum(axis=1)
        dist = sqSum**0.5
        #升序排序
        sortedIndices = dist.argsort()
        #取出最小的距离，即找到最近的中心点
        idx = sortedIndices[0]
        #该中心点对应+1
        featVec[0][idx] += 1
    return featVec

# Build Bag of words
def build_centers(path,K):
    # os.listdir(path)表示在path路径下的所有文件和和文件夹列表
    cate = [path+'/'+x for x in os.listdir(path) if os.path.isdir(path+'/'+x)]
    features = np.float32([]).reshape(0, 64) # save features of images
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder+'/*.tif'):
            #print('reading the images:%s'%(im)) # im--path of image
            img = cv2.imread(im)
            # SIFT feature
            img_f = calcSiftFeature(img)
            # Add feature
            features = np.append(features, img_f, axis=0)
    print('features:', features.shape)
    # learn from BoW
    centers = learnVocabulary(features, K)
    filename = "svm/svm_centers.npy"
    np.save(filename, centers)
    print('Bag of Words', centers.shape)

# Generate word frequency file
# Calculate the feature vector of dataset
def cal_vec(path, K):
    centers = np.load("E:/WJF_LST_Project/LDA_model/LDA_demo/svm/svm_centers.npy")
    data_vec = np.float32([]).reshape(0, K)#存放训练集图片的特征
    labels = np.float32([])
    cate=[path+'/'+x for x in os.listdir(path) if os.path.isdir(path+'/'+x)]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.tif'):
            #print('reading the images:%s'%(im))
            img = cv2.imread(im)
            # SIFT feature
            img_f = calcSiftFeature(img)
            # find feature vector in Bow
            img_vec = calcFeatVec(img_f, centers, K)
            # calculate word frequency
            data_vec = np.append(data_vec, img_vec,axis=0)
            labels = np.append(labels, idx)
            #print(data_vec)
            #print(idx)
    return data_vec, labels

#SVM training
def SVM_Train(data_vec,labels):
    # set parameters
    clf = svm.SVC(decision_function_shape='ovo')
    # get parameters
    clf.fit(data_vec,labels)
    joblib.dump(clf, "svm/svm_model.m")

#SVM test
def SVM_Test(path,K):
    # load svm model
    clf = joblib.load("svm/svm_model.m")
    # load centers
    centers = np.load("svm/svm_centers.npy")
    # cal vector
    data_vec,labels = cal_vec(path,K)
    #lda = joblib.load('saved_model/LDA_sklearn_main.model')
    #lda_vec=lda.fit_transform(data_vec)
    res = clf.predict(data_vec)
    num_test = data_vec.shape[0]
    acc = 0
    for i in range(num_test):
        if labels[i] == res[i]:
            acc = acc+1
    return acc/num_test,res

def n_array(n):
    L = []
    for i in range(0,n):
        L.append(i)
    return L

def save_bar_graph(topic_distribution):
    plt.style.use('seaborn')
    img_num,topic_num=np.shape(topic_distribution)
    topic=n_array(topic_num)
    for i in range(img_num):
         fig = plt.figure()
         y = topic_distribution[i]
         x = topic
         plt.bar(x, y,color='grey',label='image%d' % i)
         plt.legend()

         x_ticks = np.arange(0,topic_num,1)
         plt.xticks(x_ticks)
         y_ticks = np.arange(0,1,0.1)
         plt.xlabel('topic')
         plt.ylabel('probability distribution')
         plt.yticks(y_ticks)
         plt.title(u'Topic distribution of Image %d' % i, FontProperties=font)
         plt.savefig('lda_plot/lda_result_%d.jpg' % i)
    #plt.show()

def save_word_graph(word_distribution):
    plt.style.use('seaborn')
    img_num,topic_num=np.shape(word_distribution)
    topic=n_array(topic_num)
    for i in range(img_num):
         fig = plt.figure()
         y = word_distribution[i]
         x = topic
         plt.bar(x, y,color='darkcyan',label='topic %d' % i)
         plt.legend()

         x_ticks = np.arange(0,topic_num,1)
         plt.xticks(x_ticks)
         # y_ticks = np.arange(0,1,0.1)
         plt.xlabel('word')
         plt.ylabel('probability distribution')
         # plt.yticks(y_ticks)
         plt.title(u'Word distribution of Topic %d' % i, FontProperties=font)
         plt.savefig('lda_plot/topic-word/lda_result_%d.jpg' % i)

def classify(topic_distribution,path):
    data_vec = np.float32([]).reshape(0, 33)  # 存放训练集图片的特征
    labels = np.float32([])
    img_name = []
    img_num, topic_num = np.shape(topic_distribution)
    cate = [path + '/' + x for x in os.listdir(path) if os.path.isdir(path + '/' + x)]
    for idx, folder in enumerate(cate):
        imglist=glob.glob(folder + '/*.tif')
        img_topiclist=n_array(img_num)
        for i,im in zip(img_topiclist,imglist):
             max_index, max_number = max(enumerate(topic_distribution[i]), key=operator.itemgetter(1))
             img_name = np.append(img_name,im)
             labels = np.append(labels, max_index)
    img_name_list = img_name.tolist()
    labels_list = labels.tolist()
    # category_list=np.concatenate((img_name_list,labels_list),axis=2)
    for i in range(topic_num):
        os.mkdir('./lda_classify/'+str(i))
    j = 0
    for i in range(img_num):
        label = int(labels[i])
        shutil.copy(str(img_name[j]), './lda_classify/'+str(label)+'/'+str(j)+'.jpg')
        j += 1

if __name__ == "__main__":

    start = time.process_time()
    """
    # 1.LDA model training
    # API: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
    # print word frequency matrix
    """
    input_dir = r'data/feature vector/word_frequency_file.txt'
    #output_dir = r'Feature vector/poi_density_tfidf.txt'
    data_mat = np.loadtxt(input_dir)
    # LDA model
    lda = LatentDirichletAllocation(n_components=600,# document vector
                                doc_topic_prior=0.85,# alpha
                                max_iter = 2000,
                                learning_method='online',
                                verbose=True)

    lda_topic = lda.fit_transform(data_mat)
    print(lda.perplexity(data_mat))  # 收敛效果
    joblib.dump(lda, 'lda_saved_model/LDA_sklearn_main.model')

    # print doc-topic distribution
    np.set_printoptions(suppress=True)
    print(lda_topic)
    np.savetxt("lda_result/doc_topic_distribution_600_2k.txt", lda_topic, fmt='%6f', delimiter=' ')
    # print topic-word distribution
    print(lda.components_)
    np.savetxt("lda_result/topic_word_distribution_600_2k.txt", lda.components_, fmt='%6f', delimiter=' ')

    """
        # 2.plot LDA results
        # Topic_word_distribution
        # Doc_topic_distribution
    """
    # plot the distribution graph
    save_bar_graph(lda_topic)
    save_word_graph(lda.components_)
    # classify(lda_topic, train_path)
    end = time.process_time()

    print('process time is % 6.3f' %(end-start))

    """
    # SVM Classifier
    SVM_Train(lda_topic, y_train)
    #SVM_Train(x_train,y_train)
    elapsed = (time.process_time() - start)
    print('training time：',elapsed)
    print(x_train.shape)
    print(y_train)
    # Accuracy
    acc,res = SVM_Test(test_path,K)
    print('Test accuracy：',acc)
    """
# ===============================================================================