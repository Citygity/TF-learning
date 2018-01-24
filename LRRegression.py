# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:23:37 2018
首先预处理数据，例如删除无效数据，对数据进行归一化等，然后训练模型，最后测试结果
@author: rayshea
"""

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

column_names=['sample code number','1','2','3','4','5','6','7','8','9','class']
#利用pandas从网上下载数据
data=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=column_names)
#删除丢失的不完整的数据
data=data.replace(to_replace='?',value=np.nan)
data=data.dropna(how='any')

x_train,x_test,y_train,y_test=train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)

ss=StandardScaler()
#标准化数据，保证每个纬度的特征数据方差为1，均值为0->使得预测结果不会被某些纬度过大的特征值而主导。
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)

lr=LogisticRegression()
sgdc=SGDClassifier()
lr.fit(x_train,y_train)
lr_y_predict=lr.predict(x_test)

sgdc.fit(x_train,y_train)
sgdc_y_predict=sgdc.predict(x_test)
print("Accuracy of LR Classifier:",lr.score(x_test,y_test))
print(classification_report(y_test,lr_y_predict,target_names=['Benign','Malignant']))
print("finish")
print('---------------------------------------')
print("Accuracy of LR Classifier:",sgdc.score(x_test,y_test))
print(classification_report(y_test,sgdc_y_predict,target_names=['Benign','Malignant']))
print("finish")