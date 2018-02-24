from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 4)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)
yPredictions = knn.predict(X_test)
score = metrics.accuracy_score(y_test,yPredictions)
print (score)

# Notice that different values for random state provide different scores such
# 95% 97% and 100%. This is why testing accuracy is known as a high variance
# estimate

# What if created a bunch of train/test splits, calculated the testing accuracy
# for each and averaged the results together? This is cross validation!
# This solves the high variance issue

# Pretend that we have a dataset with 25 observations
# Suppose that we would like to execute 5 fold cross validation
kf = KFold(25,n_folds = 5,shuffle = False)

# Cross validation essentially provides a more accurate estimate of
# out of sample accuracy. It also uses the data more efficiently then
# train/test split since every observation is used for both training and
# testing the model

'''
Advantages of train/test split:
1. Runs K times faster than K fold cross validation
2. Simpler to examine the detailed results of the testing process

Cross validation recommendations:
1. K can be any number but K = 10 is generally recommended
2. For classification problems: Stratified sampling is recommended for creating
   the folds. For example: If the responses are A and B and 20% of the
   observations are A then about 20% of the observations in the
   cross validation folds should be A
'''

# Goal: Select the best tuning parameters (also known as hyperparameters) for
# KNN on the iris dataset

# 10 fold cross validation with K = 5 for KNN
# Note that stratified sampling is done by default
# Note that cv = 10 for 10 fold cross validation
# We are using classification accuracy as the evaluation metric
# Therefore we use scoring = "accuracy"
knn = KNeighborsClassifier(n_neighbors = 5)
scores = cross_val_score(knn,X,y,cv = 10,scoring = "accuracy")
print (scores)

# The cross_val_score method essentially completes steps 1 to 4 for
# K fold cross validation in the data school document

# Use the average accuracy as an estimate of out of sample accuracy
print (scores.mean())

# Search for an optimal value of K for KNN
k_range = range(1,31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn,X,y,cv = 10,scoring = "accuracy")
    k_scores.append(scores.mean())
print (k_scores)

# Plot the the cross validated accuracy versus the value of K for KNN
plt.plot(k_range,k_scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Cross Validated Accuracy")
plt.show()

# When deciding which value of K to use: It is generally recommended to choose
# the value that produces the simplest model
# In the case of KNN: Higher values of K produce lower complexity models
# Therefore we will choose K = 20

# Goal: Compare the best KNN model with logistic regression on the iris dataset
knn = KNeighborsClassifier(n_neighbors = 20)
score = cross_val_score(knn,X,y,cv = 10,scoring = "accuracy").mean()
print (score)

logreg = LogisticRegression()
score = cross_val_score(logreg,X,y,cv = 10,scoring = "accuracy").mean()
print (score)

# We conclude that KNN is likely a better choice than logistic regression

# Goal: Conduct feature selection by determing whether the Newspaper feature
# should be included in the linear regression model on the advertising dataset
data = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv",index_col = 0)
feature_cols = ["TV","Radio","Newspaper"]
X = data[feature_cols]
y = data["Sales"]
lm = LinearRegression()
scores = cross_val_score(lm,X,y,cv = 10,scoring = "mean_squared_error")
print (scores)

# Notice that all of the value are negative (this would not make sense
# since we are squaring all of the differences).
# This was a design decision: All values were negated on purpose
# This is because we normally look for the highest value when trying to
# optimize accuracy (for example: The accuracy metric)
# But with MSE: We are looking for the lowest value as we are trying to
# minimize it
# Negating all of the values allows us to always ask for the highest value
# When functions call cross_val_score: Those functions can assume that higher
# results indicate better models

# Fix the signs of the MSE scores
mse_scores = -scores
print (mse_scores)

# Convert MSE to RMSE
rmse_scores = np.sqrt(mse_scores)
print (rmse_scores)
print (rmse_scores.mean())

# Now let us try removing the Newspaper feature
feature_cols = ["TV","Radio"]
X = data[feature_cols]
score = np.sqrt(-cross_val_score(lm,X,y,cv = 10,scoring = "mean_squared_error")).mean()
print (score)

# We therefore conclude that the model excluding Newspaper is a better model
