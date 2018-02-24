from sklearn.datasets import load_iris

'''
Iris data set uses 4 features:
Sepal Length
Sepal Width
Petal Length
Petal Width
Each set of 4 values is associated with one of three species
Example Datapoint: (5.1,3.5,1.4,0.2,Iris = setosa)

This is a supervised learning problem in which we are attempting to predict the
species using the available data. This is supervised learning because we are trying
to learn the relationship between the data and the outcome (species).

If this was unlabelled data meaning that we only had the measurements but not the
species: This is unsupervised learning. We would attempt to cluster the samples into
meaningful groups.
'''

# This object is a special container called a bunch which is scikit's special object
# type for storing datasets and their attributes
iris = load_iris()

# One of these attributes is called data
# Each row represents one flower
print (iris.data)

# Each row is an observation
# Equivalent terms for observation: sample example instance record
# Each column is a feature
# Equivalent terms for feature: predictor attribute independent variable inout
#                               regressor covariate

print (iris.feature_names)

# Print integers representing species of each observation
print (iris.target)
print (iris.target_names)

# Equivalent terms for target: response outcome label dependent variable

'''
Two Types of Supervised Learning
Classification: Response is categorical
Regression: Response is ordered and continuous
'''

'''
In scikit-learn:
Data and targets must be stored separately and numerically
Shape of data: first dimension = number of observations
               second dimension = number of features
Shape of response: single dimension matching number of observations
'''

print (type(iris.data))
print (type(iris.target))
print (iris.data.shape)
print (iris.target.shape)

# Store feature matrix as X
X = iris.data

# Store respons vector in y
y = iris.target

'''
K nearest neighbours (KNN) Classification
1. Pick a value for K
2. Search for the K observations in the training data that are closest to the
   measurements of the unknown iris
3. Use the most popular response value (tally them) from the K nearest
   neighbours as the predicted response value for the unknown iris
'''
