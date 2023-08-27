# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: GRACIA RAVI R
RegisterNumber: 212222040047
/*

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

*/
```

## Output:
![s1](https://github.com/gracia55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/129026838/02f69083-f4a4-4926-9e5d-67cca77331cb)
![s2](https://github.com/gracia55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/129026838/cff6b37b-4cb9-4d3b-bb42-5a5b8f696391)
![s3](https://github.com/gracia55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/129026838/77c3fe0c-c85d-4807-8368-4fdf757636dd)
![s4](https://github.com/gracia55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/129026838/1d587fea-8dca-46d1-af25-9d8d9f950c7b)
![s5](https://github.com/gracia55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/129026838/1db92c28-a6cc-4eb2-a6bc-60b8924cde61)
![s6](https://github.com/gracia55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/129026838/18a9a392-7fe0-484f-9709-3cc2ea2545e6)
![s7](https://github.com/gracia55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/129026838/974bd956-6406-4aad-99ee-b3a92d04f8b5)
![s8](https://github.com/gracia55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/129026838/9663ee35-0d77-4533-b64b-0fc0dc58e078)








## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
