# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:19:17 2019

@author: 37112
"""

from keras.layers import Embedding, Dense, LSTM, Dropout
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer #  每个单词都由一个唯一的整数来表示
from keras.preprocessing.sequence import pad_sequences
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from keras.models import load_model

# 导入数据
#df = pd.read_csv("data/train.csv", names=['label','text'])

data = pd.read_csv('data/train.csv', names = ['label','text'])
data1 = pd.read_csv("data/test.csv", names = ['id','text'])
texts, classes, test = [], [], []

for i ,label in enumerate(data['label'][1:]):
    texts.append(data['text'][i+1])
    if label == 'ham':
        classes.append(0)
    else:
        classes.append(1)
n = len(texts)       
 
for i in data1['text'][1:]:
    test.append(i)
    texts.append(i)
    
texts = np.asarray(texts)
classes = np.asarray(classes)
test = np.asarray(test)

#test_x = df1['text'][1:]
maxFeatures = 10000 # 特征词
maxLen = 500 # 文档长度

# 数据预处理， 向量化
#train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)
# =============================================================================
# trainingData = int(len(texts) * 0.8)
# validationData = int(len(texts)-trainingData)
# =============================================================================
tokenizer = Tokenizer() # 每个单词都由一个唯一的整数来表示
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index # 不同词汇的个数
data = pad_sequences(sequences, maxlen=maxLen)
# =============================================================================
# print(data.shape) # (5572, 500)
# indices = np.arange(data.shape[0]) # (5572,)
# np.random.shuffle(indices) 
# data = data[indices]
# labels = classes[indices]
# x_train = data[:trainingData]
# y_train = labels[:trainingData]
# x_test = data[trainingData:]
# y_test = labels[trainingData:]
# =============================================================================
x_train = data[:n]
y_train = classes
x_test = pad_sequences(tokenizer.texts_to_sequences(test), maxlen=maxLen)
# 构建模型 LSTM
model = Sequential()
# 词向量的长度---> 嵌入到32维的向量空间中
model.add(Embedding(maxFeatures, 50))# 嵌入层将正整数（下标）转换为具有固定大小的向量  
# 比如输入为[batch,500] 其中值最大不超过maxFeatures，输入为[batch,500,32]
model.add(Dropout(0.3))
model.add(LSTM(50))
model.add(Dense(1, activation = 'relu'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.fit(x_train, y_train, epochs = 20, batch_size=60)
model.fit(x_train, y_train, epochs=20, batch_size=60, validation_split=0.2)
model.save('data/model.h5')
category_map = {0:'ham', 1:'spam'}
predicted = model.predict_classes(x_test)
#print(confusion_matrix(predicted, y_test))
np.savetxt("data/res.csv", predicted, delimiter=',', fmt = '%s')

#print('Accuracy score: ', format(accuracy_score(y_test, predicted)))
#print('Precision score: ', format(precision_score(test_y, predicted)))
#print('Recall score: ', format(recall_score(test_y, predicted)))
#print('F1 score: ', format(f1_score(test_y, predicted)))

 