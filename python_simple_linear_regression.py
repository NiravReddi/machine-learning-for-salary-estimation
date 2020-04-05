# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 20:02:16 2020

@author: MY PC
"""


import pandas as pd
import numpy as np
from matplotlib import pyplot as mt

#importing dataset
dataset=pd.read_csv("Salary_Data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
#missing values
#categorising data
#feature scaling
#spliting datasets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)
#linear reggression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

mt.scatter(x_train,y_train)
mt.plot(x_train,regressor.predict(x_train))
mt.show()