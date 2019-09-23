# article_blog
## sensitive
存储了反动词典和暴恐词典
## preprocessing
blog.py 对原始数据的预处理，原始数据为44W篇文章，其中包括1000+异常文章，去掉html标签提取其文本，并利用jieba进行文章分词，用DFA进行敏感词过滤，将结果存入数据库
same_blog.py 将同一用户的文章分词以及敏感词匹配结果结合到一起，存入数据库，一共为20733个用户，
## 其他
本项目是针对用户群体进行的聚类
user_dbscan.py 针对用户进行聚类，采用PCA降维，最后加上敏感词数量作为一个维度，得到聚类结果
csv_to_mongo.py 将分类结果导入数据库
label.csv 得到的聚类结果

## LDA_cluster.py
基于LDA对文章进行主题提取的聚类，还计算了perplexity选取最优topic，pyLDAvic库进行聚类可视化
