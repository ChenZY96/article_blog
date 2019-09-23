#coding=utf-8

import gevent
from gevent.threadpool import ThreadPool
from pymongo import MongoClient
from pyquery import PyQuery as pq
import jieba.analyse
import re
import DFAfilter

client = MongoClient(host='192.168.58.59', port=27017)
db = client.dataset_release
collection = db.blog

article_db = db.article  # 处理后的blog数据库

f = DFAfilter.DFAFilter()
path = "sensitive/fandong.txt"
path2 = "sensitive/baokong.txt"
f.parse(path)
f.parse(path2)

count = 0


def article(each):
    t = pq(each['content_html'])
    title = pq(each['title_html'])

    t.find('script').remove()
    title.find('script').remove()

    content = t.text()  # 文章内容
    title_text = title.text()
    title = title_text.split('\n')[0]  # 标题
    #print(title)
    return title,content


def fenci(content):
    # 停用词
    stopwords_list = [line.strip() for line in open('stopwords.txt', encoding='UTF-8').readlines()]

    # 数据处理
    content = content.replace('撸了今年阿里、头条和美团的面试，我有一个重要发现.......>>>', '')
    content = content.replace('\n', '')
    # new_content = re.sub("[A-Za-z0-9\:\·\—\，\。\“ \”]", '', content)
    new_content = re.sub("[\:\·\—\，\。\“ \”\.]", '', content)

    # jieba分词
    seg_list = jieba.cut(new_content, cut_all=False)
    cut_words = ','.join(seg_list)
    res_cut_words = list(filter(lambda x: x not in stopwords_list and len(x) > 1, cut_words.split(',')))

    # 关键词提取
    jieba.analyse.set_stop_words("stopwords.txt")
    keywords = jieba.analyse.extract_tags(new_content, topK=10, withWeight=True)
    #print(keywords)
    #print(res_cut_words)
    return res_cut_words,keywords


def iteration(each):

    uid = each['uid']
    if each.get('content_html', False):
        title, content = article(each)  # 文章处理
        res_cut_words, keywords = fenci(content)  # 分词，关键词
        sen_words, sen_words_count = f.count_sen(content)  # 敏感词

        global count
        if count % 1000 == 0:
            print(count)
        count += 1

        data = {'uid': uid, 'title': title, 'content': content, 'cut_words': res_cut_words, \
                'keywords10': keywords, 'sen_words_count': sen_words_count}

        article_db.insert_one(data)

        # data_list.append(data)
        #
        # count += 1
        #
        # if count % 10 == 0:
        #     print(count)
        #     print(data_list)
        #     # article_db.insert_many(data_list)
        #     data_list = []


def main():

    all_content = collection.find({}, {'uid': 1, 'title_html': 1, 'content_html': 1}, no_cursor_timeout=True)

    pool = ThreadPool(200)
    threads = [pool.spawn(iteration, each) for each in all_content]
    gevent.joinall(threads)



if __name__ == '__main__':

    main()







