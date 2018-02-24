from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris = load_iris()

X = iris.data
y = iris.target

print (X.shape)
print (y.shape)

# Instantiate the estimator (make an instance of the KNN model)
# Notice that we choose K = 1
knn = KNeighborsClassifier(n_neighbors = 1)

# Printing knn will show the default values for all of the unspecified
# parameters
print (knn)

# Fitting the data
knn.fit(X,y)

# Predicting the response for a new observation
newDataPoint = np.array([[3,5,4,2]])
response = knn.predict(newDataPoint)
print (response)

# The predict method can act on multiple arrays at once
newDataPoints = np.array([[3,5,4,2],[5,4,3,2]])
responses = knn.predict(newDataPoints)
print (responses)

# We can try tuning the model by using K = 5
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X,y)
responses = knn.predict(newDataPoints)
print (responses)
