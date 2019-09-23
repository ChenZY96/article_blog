#coding=utf-8
from pymongo import MongoClient
from csv import DictReader
import pandas as pd

# 连接数据库
client = MongoClient(host='192.168.58.59', port=27017)
db = client.single_community_dataset
blog = db.blog


with open('label.csv',mode='r',encoding='utf-8') as f:
    data = DictReader(f)
    blog.insert_many(data)
print('Import to Mongo successfully')
    

