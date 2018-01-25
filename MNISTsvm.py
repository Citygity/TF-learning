# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 21:34:47 2018
该数据为8*8的数字集合，测试集大小为450，训练集大小为1347
@author: rayshea
"""

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
digits=load_digits()

x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.25,random_state=33)

ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)

lsvc=LinearSVC()
#初始化svm分类器
lsvc.fit(x_train,y_train)
#训练分类器
y_predict=lsvc.predict(x_test)
print('The Accuracy of Linear SVC is',lsvc.score(x_test,y_test))
print(classification_report(y_test,y_predict,target_names=digits.target_names.astype(str)))
