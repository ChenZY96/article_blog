#coding=utf-8
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
import pyLDAvis
import pyLDAvis.sklearn
import re
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import Counter
from bson import ObjectId
import numpy as np


def cut_words_list(cut_words):
    data = ' '.join(each for each in cut_words)
    data = re.sub("[A-Za-z]", '', data)
    return data

def print_top_words(model,feature_names,n_top_words):
    for topic_idx,topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

#生成停用词
with open('stopwords.txt',mode='r',encoding='utf-8') as f:
    stoplist = [each.strip() for each in f.readlines()]

#连接数据库
client = MongoClient(host='192.168.58.59', port=27017)
db = client.dataset_release
article = db.article


all_articles = article.find({}, {'uid': 1, 'title': 1, 'cut_words': 1}, no_cursor_timeout=True)

#用TfidfVectorizer可能效果更好
tf_vectorizer = CountVectorizer(strip_accents='unicode',max_features=4000,max_df=0.2, min_df=5,)

#生成语料库
corpus_path = 'corpus.pkl'
if os.path.exists(corpus_path):
    corpus = joblib.load(corpus_path)
    print('corpus has done')
else:
    corpus = [cut_words_list(stoplist,each['cut_words_count'])for each in all_articles]
    joblib.dump(corpus,corpus_path)
    print('corpus has saved')
corpus = [cut_words_list(each['cut_words']) for each in all_articles]

# 存储tf模型
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


# weight = tf.toarray()
# print(weight.shape)
# 存储初始矩阵，方便后续调参
src_weight = 'weight.pkl'
if os.path.exists(src_weight):
    weight = joblib.load(src_weight)
else:
    weight = tf.toarray()
    joblib.dump(weight,src_weight)
print(weight.shape)

# 指定Topic数量
n_topics = 50
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50, learning_method='online', learning_offset=50., random_state=0)
lda.fit(tf)

# 分类结果
topic_dist = lda.transform(tf)
labels = [-1]*len(topic_dist)
i = 0
for dist in topic_dist:
    dist = dist.tolist()
    if len(set(dist)) != 1:
        review_topic_index = dist.index(max(dist))
        review_max_topic_probality = dist[review_topic_index]
        labels[i]=review_topic_index
        i = i + 1
    else:
        i = i + 1
#print(labels)
# 统计每个分类的数量
print(Counter(labels))

# 分类结果存入csv
all_articles = article.find({}, {'uid': 1, 'title': 1, 'cut_words': 1}, no_cursor_timeout=True)
uid = []
title = []
labels = []
for id,each in enumerate(all_articles):
    uid.append(each['uid'])
    title.append(each['title'])
    labels.append(labels[id])
df = pd.DataFrame()
df['uid'] = uid
df['title'] = title
df['label'] = labels
df.to_csv('label.csv',encoding='utf-8_sig')


# 模型的保存
joblib.dump(lda,'lda_50.m')
# # 模型的读取
# lda = joblib.load('lda_50.m')

n_top_words = 20
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda,tf_feature_names,n_top_words)

# 可视化
#pyLDAvis.enable_notebook()
data = pyLDAvis.sklearn.prepare(lda,tf,tf_vectorizer)
pyLDAvis.show(data)

# 计算perplexity
grid = dict()
for i in range(1,100,5):
    grid[i] = list()
    n_topics = i
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=50, learning_method='online', learning_offset=50.,
                                    random_state=0)
                                    lda.fit(tf)
                                    train_perplexity = lda.perplexity(tf)
                                    print('sklearn preplexity: train=%.3f' % (train_perplexity))
                                    grid[i].append(train_perplexity)

# 画图展示困惑度
df = pd.DataFrame(grid)
df.to_csv('sklearn_perplexity.csv')
plt.figure(figsize=(14, 8), dpi=120)
# plt.subplot(221)
plt.plot(df.columns.values, df.iloc[0].values, '#007A99')
plt.xticks(df.columns.values)
plt.ylabel('train Perplexity')
plt.show()
plt.savefig('lda_topic_perplexity.png', bbox_inches='tight', pad_inches=0.1)









