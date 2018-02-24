from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
y = iris.target

'''
How do I choose which model to use?
How do I choose the best tuning parameters for that model?
How do I estimate the likely performance of my model on out of sample data?
'''

'''
Evaluation Procedure 1:
1. Train the model on the entire dataset.
2. Test the model on the same dataset and evlauate how well we did by comparing
   predicted response values with the true response values
'''

# We will try logisitic regression first
logreg = LogisticRegression()
logreg.fit(X,y)
logreg.predict(X)

# Store the response values
yPredictions = logreg.predict(X)

# We then evaluate the training accuracy
# Classification accuracy is the proportion of correct predictions
score = metrics.accuracy_score(y,yPredictions)
print (score)

# We can then repeat this procedure for KNN with K = 5 and K = 1
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X,y)
yPredictions = knn.predict(X)
score = metrics.accuracy_score(y,yPredictions)
print (score)

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X,y)
yPredictions = knn.predict(X)
score = metrics.accuracy_score(y,yPredictions)
print (score)

# We conclude the KNN with K = 1 is the best model to use with this data
# Note that obtaining 100 % accuracy with K = 1 is expected since we are testing
# the training data
# We therefore conclude that training and testing our models on the exact same data
# is not a useful procedure for deciding which models to choose
# Our goal here is to estimate how well each model is likely to perform on
# out of sample data
# If what we try to maximize is training accuracy: Then we are rewarding models
# that will not necessarily generalize to future data
# Creating an overly complex model is known as overfitting

'''
Evaluation Procedure 2: Train Test Split
1. Split the dataset into two pieces: A training set and a testing set
2. Train the model on the training set
3. Test the model on the testing set and evaluate how well we did
We are therefore simulating how likely the model is to perform well on
out of sample data
'''

# Note that if we do not call random state then the data that is selected for testing
# will be random every time
# Someone else using random state = 4 will have the data selected for testing as us
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state = 4)

# We can now train different models on the training set

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
yPredictions = logreg.predict(X_test)
score = metrics.accuracy_score(y_test,yPredictions)
print (score)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)
yPredictions = knn.predict(X_test)
score = metrics.accuracy_score(y_test,yPredictions)
print (score)

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train,y_train)
yPredictions = knn.predict(X_test)
score = metrics.accuracy_score(y_test,yPredictions)
print (score)

# Can we locate an even better value of K?
kValues = range(1,26)
scores = []
for k in kValues:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    yPredictions = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,yPredictions))
plt.plot(kValues,scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Testing Accuracy")
plt.show()

# We see that K = 6 to K = 17 produces the highest testing accuracy
# We will choose K = 11 because it is in the middle of the range
# Because this dataset is so small: It is hard to reliably say that the
# behaviour we are seeing will indeed generalize

knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit(X,y)
response = knn.predict([3,5,4,2])
print (response)

'''
The disadvantage of train/test split is that it provides a high variance on
out of sample accuracy (it can change a lot depending upon which observations
happen to be in the training set versus the testing set).
K fold cross validation overcomes this limitation by repeating the train/test
split process multiple times.
But train/test split is still useful because of its flexibiltiy and speed.
'''
