# Building My First Machine Learning Model

#1. Loading Datasets as Dataframes

import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')

#2. Data Processing i.e Assigning Variables
X = df.drop(['logS'], axis=1)
Y = df.logS

#3. Data Spliting, Test&Train 80/20

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Build Model
#LinearRegression

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)

#predict
Y_lr_train_pred = lr.predict(X_train)
Y_lr_test_pred = lr.predict(X_test)

#perfomance metrics using mean square error and r2score

from sklearn.metrics import mean_squared_error, r2_score
lr_train_mse = mean_squared_error(Y_train, Y_lr_train_pred)
lr_train_r2 = r2_score(Y_train, Y_lr_train_pred)
lr_test_mse = mean_squared_error(Y_test, Y_lr_test_pred)
lr_test_r2 = r2_score(Y_test, Y_lr_test_pred)

#results

lr_results = pd.DataFrame(['Linear regression',lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']

print(lr_results)

#Visualising through a Graph

import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(5,5))
plt.scatter(x=Y_train, y=Y_lr_train_pred, c="#7CAE00", alpha=0.3)
z = np.polyfit(Y_train, Y_lr_train_pred, 1)
p = np.poly1d(z)
plt.plot(Y_train,p(Y_train),"#F8766D")
plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')
plt.show()