# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:19:17 2019

@author: 37112
"""

from keras.layers import SimpleRNN, Embedding,Dense, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier as abc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nltk.stem.porter import PorterStemmer
# glove 句子向量转化
def get_words_vec(q, emb, vocab):
    glove_index_list = []
    # 取句子对应单词的索引
    for word in q.split():
        if word in vocab:
            index = vocab.index(word)
            glove_index_list.append(index)
    return emb[glove_index_list].astype(float).sum()/len(q.split())



# 导入数据
df = pd.read_csv("data/train.csv", names=['label','text'])
df1 = pd.read_csv("data/test.csv", names=['id','text'])
train_x = df['text'][1:]
train_y = df['label'][1:]
test_x = df1['text'][1:]


# 数据预处理， 向量化
#train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)

# =============================================================================
# # stem
# p = PorterStemmer()
# for i in range(len(train_x)):
#     train_x_vec = ""
#     line = train_x.iloc[i]
#     for j in line.split():
#         j = p.stem(j)
#         train_x_vec += j
#     train_x.iloc[i] = train_x_vec
# 
# for i in range(len(test_x)):
#     test_x_vec = ""
#     line = test_x.iloc[i]
#     for j in line.split():
#         j = p.stem(j)
#         test_x_vec += j
#     test_x.iloc[i] = test_x_vec
# =============================================================================

    
# TFIDF方法
vectorizer = TfidfVectorizer()
x_train_tfidf = vectorizer.fit_transform(train_x)
x_input_tfidf = vectorizer.transform(test_x)
# 构建模型 朴素贝叶斯
model = abc(n_estimators = 600)
model.fit(x_train_tfidf, train_y)

#
predicted = model.predict(x_input_tfidf)
#print(confusion_matrix(predicted, test_y))
np.savetxt("data/res.csv", predicted, delimiter=',', fmt = '%s')

#print('Accuracy score: ', format(accuracy_score(test_y, predicted)))
#print('Precision score: ', format(precision_score(test_y, predicted)))
#print('Recall score: ', format(recall_score(test_y, predicted)))
#print('F1 score: ', format(f1_score(test_y, predicted)))

 
# =============================================================================
# #output some examples
# category_map = {'ham':0, 'spam':1}
# for sentence, category, real in zip(test_x[:10], predicted[:10], test_y[:10]):
#     print('\nmessage_content:', sentence, '\npredicted_type:', category_map[category], 'real_values:', category_map[real])
# 
# =============================================================================
