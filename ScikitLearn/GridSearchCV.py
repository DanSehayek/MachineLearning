from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt

# Goal: Select the best tuning parameters for KNN on the iris dataset

iris = load_iris()
X = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors = 5)
scores = cross_val_score(knn,X,y,cv = 10,scoring = "accuracy")
print (scores)
print (scores.mean())

k_range = range(1,31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn,X,y,cv = 10,scoring = "accuracy")
    k_scores.append(scores.mean())
print (k_scores)

plt.plot(k_range,k_scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Cross Validated Accuracy")
plt.show()

'''
The process above can be done more efficiently using GridSearchCV
'''

from sklearn.grid_search import GridSearchCV

# Define the parameter values to be searched
k_range = list(range(1,31))

# Create a parameter grid: Map the parameter names to the values that
# should be searched
param_grid = dict(n_neighbors = k_range)
print (param_grid)

# Instantiate the grid
grid = GridSearchCV(knn,param_grid,cv = 10,scoring = "accuracy")

# The grid is an object that is ready to do 10 fold cross validation
# on a KNN model using classification accuracy as the evaluation metric
# Note that setting n_jobs = -1 will instruct the sklearn to use all
# available processors if parallel processing is supported

# Fit the grid with data
grid.fit(X,y)

# View the complete results
results = grid.grid_scores_
print (results)

# There is one tuple for each of the 30 trials of cross validation
# Note that if the standard deviation is high then the cross validated
# estimate of the accuracy may not be as reliable

# Examine the first tuple
print (grid.grid_scores_[0].parameters)
print (grid.grid_scores_[0].cv_validation_scores)
print (grid.grid_scores_[0].mean_validation_score)

# Create a list of the mean scores
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print (grid_mean_scores)

# Plot the results
plt.plot(k_range,grid_mean_scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Cross Validated Accuracy")
plt.show()

# Examine the best model
print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)

'''
Searching multiple parameters simultaneously
'''

# Another parameter that we can tune is the weights parameter which controls
# how the K nearest neighbours are weighted when making a prediction
# The default option is uniform which means that all points in the
# neighourhood are weighed equally
# Another option is distance which weights closer neighbours more heavily than
# further neighbours

k_range = list(range(1,31))
weight_options = ["uniform","distance"]

param_grid = dict(n_neighbors = k_range,weights = weight_options)
print (param_grid)

grid = GridSearchCV(knn,param_grid,cv = 10,scoring = "accuracy")
grid.fit(X,y)

results = grid.grid_scores_
print (results)

# Examine the best model
print (grid.best_score_)
print (grid.best_params_)

# Using the best parameters to make predictions
knn = KNeighborsClassifier(n_neighbors = 13,weights = "uniform")
knn.fit(X,y)
result = knn.predict([3,5,4,2])
print (result)

# Alternatively we can use GridSearchCV as it automatically refits the
# best model using all of the data
result = grid.predict([3,5,4,2])
print (result)

'''
RandomizedSearchCV helps to reduce computational expense
Searching many different parameters at once may be computationally infeasible
RandomizedSearchCV searches a subset of these parameters and allows you to
control the computational budget
'''

from sklearn.grid_search import RandomizedSearchCV

# Create a parameter distribution
param_dist = dict(n_neighbors = k_range,weights = weight_options)

# Note that if any of the parameters are not discrete:
# A continuous distribution must be specified for those parameters

# n_iter controls the number of random combinations it will try
# random_state = 5 is set for the purpose of reproducibility
rand = RandomizedSearchCV(knn,param_dist,cv = 10,scoring = "accuracy",n_iter = 10,random_state = 5)
rand.fit(X,y)
results = rand.grid_scores_
print (results)

# Examine the best model
print (rand.best_score_)
print (rand.best_params_)

# Note that quite often RandomizedSearchCV will find the best result
# (or something very close) in a fraction of the time that GridSearchCV
# requires

# We see from below that most of the time it does result in a score of 0.98
# Even when it does not find that score: It is still close to 0.98
best_scores = []
for i in range(20):
    rand = RandomizedSearchCV(knn,param_dist,cv = 10, scoring = "accuracy",n_iter = 10)
    rand.fit(X,y)
    best_scores.append(round(rand.best_score_, 3))
print (best_scores)

# General Recommendations: Set n_iter to a reasonable value and determine how
# long this takes. Then use this to determine the runtimes for greater values
# of n_iter and set the value of n_iter according to the maximum amount of
# computational runtime you are willing to have
