from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

iris = load_iris()

X = iris.data
y = iris.target

logreg = LogisticRegression()
logreg.fit(X,y)

newDataPoints = np.array([[3,5,4,2],[5,4,3,2]])
responses = logreg.predict(newDataPoints)
print (responses)
