#coding=utf-8
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
import joblib
from sklearn.cluster import DBSCAN
import pandas as pd
import datetime
import numpy as np
from bson import ObjectId
from sklearn.decomposition import PCA
import os
from collections import Counter

def cut_words_list(stoplist,cut_words):
    res = []
    for each in cut_words:
        if cut_words[each]>3 and each not in stoplist:
            for i in range(cut_words[each]):
                res.append(each)
    data = ' '.join(res)
    #data = re.sub("[A-Za-z]", '', data)
    return data

def sen_score(sen_words_count):
    sen_count = 0
    for each in sen_words_count:
        sen_count += sen_words_count[each]
    return sen_count


# 连接数据库
client = MongoClient(host='192.168.58.59', port=27017)
db = client.dataset_release
same_blog = db.same_blogs
art = db.articles

all_articles = same_blog.find({},{'uid':1,'cut_words_count':1,'sen_words_count':1},no_cursor_timeout=True)

with open('stopwords.txt',mode='r',encoding='utf-8') as f:
    stoplist = [each.strip() for each in f.readlines()]
    
tf_vectorizer = TfidfVectorizer(max_df=0.1,min_df=0.0006)

# save corpus
corpus_path = 'corpus.pkl'
#joblib.dump(corpus,corpus_path)
if os.path.exists(corpus_path):
    corpus = joblib.load(corpus_path)
    print('corpus has done')
else:
    corpus = [cut_words_list(stoplist,each['cut_words_count'])for each in all_articles]
    joblib.dump(corpus,corpus_path)
    print('corpus has saved')

#print(corpus)
#print(len(corpus))

# 计算敏感维度
all_articles = same_blog.find({},{'uid':1,'title':1,'cut_words':1,'sen_words_count':1},no_cursor_timeout=True)
sen_ratio = [sen_score(each['sen_words_count'])for each in all_articles]
max_sen = max(sen_ratio)
min_sen = min(sen_ratio)
sen_ratio = list(map(lambda x: float((x-min_sen)/(max_sen-min_sen)),sen_ratio))
#print(sen_ratio)

print(len(corpus))
#print(corpus)
#tf = tf_vectorizer.fit_transform(corpus)

# 存储模型
tf_ModelPath = 'tf_model.pkl'
if os.path.exists(tf_ModelPath):
    tf_vectorizer = joblib.load(tf_ModelPath)
    tf = tf_vectorizer.fit_transform(corpus)
    print('model has load')
else:
    tf_vectorizer = TfidfVectorizer(max_df=0.1, min_df=0.0006,) 
    tf = tf_vectorizer.fit_transform(corpus)
    joblib.dump(tf_vectorizer,tf_ModelPath)
    print('model has saved')


src_weight = 'weight.pkl'
if os.path.exists(src_weight):
    weight = joblib.load(src_weight)
else:
    weight = tf.toarray()
    joblib.dump(weight,src_weight)
print(weight.shape)

# PCA降维
pca = PCA(n_components=100)
weight = pca.fit_transform(weight)


# 增加敏感词维度
weight = np.c_[weight,sen_ratio]
print('final weight shape >>> ',weight.shape)

# dbscan
#db = DBSCAN(eps=0.08, min_samples=10)
print("[log]starting DBSCAN:{0}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
db = DBSCAN(eps=0.15, min_samples=10, metric='cosine',n_jobs=-1)
    #result = db.fit(weight)
source = db.fit_predict(weight)
label = db.labels_
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
n_clusters_ = len(set(label)) - (1 if -1 in label else 0)
print('[log]ending DBSCAN:{0}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
print("[output]clusters number:"+str(n_clusters_))
print("[output]cluster number and count:"+str(Counter(source)))


# 分类结果写入csv
all_articles = same_blog.find({}, {'uid': 1}, no_cursor_timeout=True)
uid = []
labels = []
for id,each in enumerate(all_articles):
    uid.append(each['uid'])
    #title.append(each['title'])
    labels.append(label[id])
df = pd.DataFrame()
df['uid'] = uid
df['sen_ratio'] = sen_ratio
df['label'] = labels
df.to_csv('label.csv',encoding='utf-8_sig',index=None)
print("[output]label.csv has saved")

# 统计各聚类簇以及异常用户聚类结果
data = pd.read_csv('label.csv',encoding='utf-8')
print(data.head())

client = MongoClient(host='192.168.58.59', port=27017)
db = client.dataset_release
article = db.blog

wrong_user = set()

all_articles = article.find({'generated':True}, {'uid': 1}, no_cursor_timeout=True)
for each in all_articles:
    wrong_user.add(str(each['uid']))
print('Abnormal user count: >>> ',len(wrong_user))

res = []
#all_user = ('5d6352c8ea309cc86a3de743','5d6352d4ea309cc86a3de78b','5d6352d3ea309cc86a3de77a')
for a_user in wrong_user:
    #print(a_user)
    #irint(a_user)
    tmp = data[data['uid'] == a_user]['label'].values[0]
    #print(tmp)
    res.append(tmp)
from collections import Counter
print('Abnormal cluster >>> ',Counter(res))
print('All_user cluster >>> ',data['label'].value_counts())

