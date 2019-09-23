# -*- coding:utf-8 -*-

import time
from collections import Counter
from pymongo import MongoClient
from bson import ObjectId


time1 = time.time()


# DFA算法
class DFAFilter(object):
    def __init__(self):
        self.keyword_chains = {}  # 关键词链表
        self.delimit = '\x00'  # 限定

    def add(self, keyword):
        keyword = keyword.lower()  # 关键词英文变为小写
        chars = keyword.strip()  # 关键字去除首尾空格和换行
        if not chars:  # 如果关键词为空直接返回
            return
        level = self.keyword_chains
        # 遍历关键字的每个字
        for i in range(len(chars)):
            # 如果这个字已经存在字符链的key中就进入其子字典
            if chars[i] in level:
                level = level[chars[i]]
            else:
                if not isinstance(level, dict):
                    break
                for j in range(i, len(chars)):
                    level[chars[j]] = {}
                    last_level, last_char = level, chars[j]
                    level = level[chars[j]]
                last_level[last_char] = {self.delimit: 0}
                break
        if i == len(chars) - 1:
            level[self.delimit] = 0

    def parse(self, path):
        with open(path, encoding='utf-8') as f:
            for keyword in f:
                self.add(str(keyword).strip())
        #print(self.keyword_chains)

    def filter(self, message, repl="*"):
        message = message.lower()
        ret = []
        start = 0
        while start < len(message):
            level = self.keyword_chains
            step_ins = 0
            for char in message[start:]:
                if char in level:
                    step_ins += 1
                    if self.delimit not in level[char]:
                        level = level[char]
                    else:
                        ret.append(repl * step_ins)
                        start += step_ins - 1
                        break
                else:
                    ret.append(message[start])
                    break
            else:
                ret.append(message[start])
            start += 1

        return ''.join(ret)

    def count_sen(self,message):
        message = message.lower()
        ret = []
        start = 0
        while start < len(message):
            level = self.keyword_chains
            step_ins = 0
            sen_word = ''
            for char in message[start:]:
                if char in level:
                    step_ins += 1
                    sen_word += char
                    if self.delimit not in level[char]:
                        level = level[char]
                    else:
                        ret.append(sen_word)
                        start += step_ins - 1
                        break
                else:
                    break
            else:
                break
            start += 1
        return ret,Counter(ret)



if __name__ == "__main__":
    client = MongoClient(host='192.168.58.59', port=27017)
    db = client.dataset_release
    collection = db.article
    sen_words_db = db.all_sen_words

    test_db = db.test

    all_content = collection.find({},{'uid':1,'article':1}).limit(10)
    #all_content = collection.find({'uid': ObjectId('5d75152add07cce828f87627')}, {'uid': 1, 'article': 1})

    gfw = DFAFilter()
    path = "sensitive/fandong.txt"
    path2 = "sensitive/baokong.txt"
    gfw.parse(path)
    gfw.parse(path2)


    for each in all_content:
        uid = each['uid']
        text = each['article']
        title = text.split(' ')[0] #文章标题
        print(title)

        #text = "你真是个打倒中共，盘古皇后，法轮功，十七大，电话交友。"
        result = gfw.filter(text)
        word,count_dict = gfw.count_sen(text)

        #print(text)
        #print(result)
        #print(word)
        print((count_dict))
        time2 = time.time()
        print('总共耗时：' + str(time2 - time1) + 's')


        test_db.update_one({'uid': each['uid']},{'$set': {'sen_word_count':count_dict}})
        #sen_words_db.insert({'uid':uid,'title':title,'sen_words':count_dict})

# 打算将敏感词存入数据库
