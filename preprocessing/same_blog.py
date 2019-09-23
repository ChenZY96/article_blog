#coding=utf-8
from pymongo import MongoClient
from collections import Counter
from bson import ObjectId
client = MongoClient(host='192.168.58.59', port=27017)
db = client.dataset_release
user_id = db.blog
article = db.article
same_blogs = db.same_blogs

all_user_cursor = user_id.find({},{'uid':1},no_cursor_timeout=True)
all_user = set()
for user in all_user_cursor:
    all_user.add(user['uid'])
#print(all_user)
print(len(all_user))
count = 0
for uid in all_user:
    count += 1
    if count % 100 == 0:
        print(count//100)
    all_articles = article.find({'uid': uid}, {'cut_words': 1,'sen_words_count':1},no_cursor_timeout=True)
    cut_words = []
    res2 = {}
    for a in all_articles:
        cut_words += a['cut_words']
        res1 = Counter(cut_words)
        if a['sen_words_count']:
            #print(e['sen_words_count'])
            for j in a['sen_words_count']:
                if j in res2:
                    res2[j] += a['sen_words_count'][j]
                else:
                    res2[j] = a['sen_words_count'][j]
    same_blogs.insert({'uid': uid, 'cut_words_count': Counter(cut_words),'sen_words_count':res2})





