# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:42:43 2016

@author: tqz828
"""

import os
import json
from pprint import pprint

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

data = pd.read_csv(r'C:\Users\tqz828\Documents\ds_bootcamp_project2\bank\bank.csv', sep =';',na_values='unknown')

data.head()


"""
Baseline Model - Logistic regression Model
"""


Y = data["y"] #Y is predictor
Y.head()

#creating a new dataframe by dropping the y column
X = data.drop("y", axis = 1, inplace = False) #inplace is to create a new dataframe. axis = 1 is column. axis = 0 is row
X.head()

X = X._get_numeric_data()
X.head()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=4)  #random state keeps the split values consistent. not explicitly required.

print len(X_train), len(X_test), len(Y_train), len(Y_test)



logreg_model = LogisticRegression()

##### Step 1: Create the model
#creating the model on training data
logreg_model.fit(X_train, Y_train)


### Step 2: Predict using the model
# using the model to predict republican or democrat
Y_predict = logreg_model.predict(X_test)
Y_predict

### Step 3: Evaluate the model
#checking the actual value in Y_test with the predicted value
accuracy_score(Y_test, Y_predict)

# precision score, recall, f1-score in one classification report
print metrics.classification_report(Y_test, Y_predict)

np.equal(Y_predict, np.array(Y_test))
