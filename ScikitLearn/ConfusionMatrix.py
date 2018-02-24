from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import binarize
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, header = None, names = col_names)

# Print the first 5 rows of data
pima.head()

# label = 0 indicates that the patient does not have diabetes
# label = 1 indicates that the patient does have diabetes

# Select the features and define X and y
feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
X = pima[feature_cols]
y = pima["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
yPredictions = logreg.predict(X_test)
score = metrics.accuracy_score(y_test, yPredictions)
print (score)

# The classification accuracy should always be compared to the null accuracy
# Null Accuracy: Accuracy that could be achieved by always predicting the most
# frequent class

# Examine the class distribution of the testing set
# (In this case: The number of 0s and the number of 1s in the dataset)
classDistribution = y_test.value_counts()
print (classDistribution)

# Calculate the percentage of ones
# (This will essentially divide the total number of ones by the
# total number of responses)
percentageOfOnes = y_test.mean()
percentageOfZeros = 1 - y_test.mean()
print (percentageOfOnes)
print (percentageOfZeros)

# Calculate the null accuracy
nullAccuracy = max(percentageOfZeros,percentageOfOnes)
print (nullAccuracy)

# This is how it would be done for binary classification problems
# For multiclass problems:
nullAccuracy = y_test.value_counts().head(1) / len(y_test)
print (nullAccuracy)

# Comparing the first 25 true and predicted responses
print ('True:', y_test.values[0:25])
print ('Pred:', yPredictions[0:25])

# The confusion matrix is a useful tool for helping to better understand
# what types of error our model is making
confusionMatrix = metrics.confusion_matrix(y_test, yPredictions)
print (confusionMatrix)

TP = confusionMatrix[1, 1]
TN = confusionMatrix[0, 0]
FP = confusionMatrix[0, 1]
FN = confusionMatrix[1, 0]

# Classification Accuracy: How often is the classifier correct?
classificationAccuracy = (TP + TN) / (TP + TN + FP + FN)
print (classificationAccuracy)
classificationAccuracy = metrics.accuracy_score(y_test, yPredictions)
print (classificationAccuracy)

# Classification Error: How often is the classifier incorrect?
classificationError = (FP + FN) / (TP + TN + FP + FN)
print (classificationError)
classificationError = 1 - metrics.accuracy_score(y_test, yPredictions)
print (classificationError)

# Sensitivity: When the actual value is positive: How often is the prediction correct?
sensitivity = TP / (TP + FN)
print (sensitivity)
sensitivity = metrics.recall_score(y_test, yPredictions)
print (sensitivity)

# Specificity: When the actual value is negative: How often is the prediction correct?
specificity = TN / (TN + FP)
print (specificity)

# False Positive Rate: When the actual value is negative: How often is the prediction incorrect?
falsePositiveRate = FP / (TN + FP)
print (falsePositiveRate)

# Precision: When a positive value is predicted: How often is the prediction correct?
precision = TP / (TP + FP)
print (precision)
precision = metrics.precision_score(y_test, yPredictions)
print (precision)

'''
Conclusion:
Confusion matrix gives you a more complete picture of how your classifier is performing
It also allows you to compute various classification metrics and these metrics
can guide your model selection
'''

# Print the first 10 predicted responses
predictedResponses = logreg.predict(X_test)[0:10]
print (predictedResponses)

# Print the first 10 predicted probabilities
predictedProbabilities = logreg.predict_proba(X_test)[0:10,:]
print (predictedProbabilities)

# Each row represents an observation
# The first entry is the predicted probability that the response is 0
# The second entry is the predicted probability that the response is 1

# Store the predicted probabilities for class 1
yPredictedProbabilities = logreg.predict_proba(X_test)[:,1]

# Plot a histogram of the predicted probabilities
plt.rcParams['font.size'] = 14
plt.hist(yPredictedProbabilities, bins = 8)
plt.xlim(0,1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')
plt.show()

# We can modify the threshold to influence factors such as sensitivity
# We will decrease the threshold to predict diabetes if the predicted
# probability is greater than 0.3
yPredictions = binarize([yPredictedProbabilities], 0.3)[0]
print (yPredictions)

# Print the first 10 predicted probabilities
print (yPredictedProbabilities[0:10])

# Print the first 10 predicted responses with the lower threshold
print (yPredictions[0:10])

# Previous confusion matrix (default threshold of 0.5)
print(confusionMatrix)

# New confusion matrix (threshold of 0.3)
newConfusionMatrix = metrics.confusion_matrix(y_test, yPredictions)
print(newConfusionMatrix)

# Sensitivity has increased (used to be 0.24)
sensitivity = 46 / (46 + 16)
print(sensitivity)

# Specificity has decreased (used to be 0.91)
specificity = 80 / (80 + 50)
print(specificity)

'''
Conclusion:
Threshold of 0.5 is used by default (for binary problems) to convert predicted probabilities into class predictions
Threshold can be adjusted to increase sensitivity or specificity
'''

'''
It would be nice if we could see how sensitivity and specificity are affected
by various thresholds without actually changing the threshold.
The answer is the ROC curve!
'''

fpr, tpr, thresholds = metrics.roc_curve(y_test, yPredictedProbabilities)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()

'''
ROC curve can help you to choose a threshold that balances sensitivity
and specificity in a way that makes sense for your particular context
You cannot actually see the thresholds used to generate the curve on the
ROC curve itself
'''

# Define a function that accepts a threshold and prints sensitivity
# and specificity
# Note that the [-1] in this case simply converts the array into a list
def evaluateThreshold(threshold):
    print("Sensitivity: {0}".format(tpr[thresholds > threshold][-1]))
    print("Specificity: {0}".format(1 - fpr[thresholds > threshold][-1]))

print (evaluateThreshold(0.5))
print (evaluateThreshold(0.3))

# AUC is the percentage of the ROC plot (the entire box) that is underneath the ROC curve
AUC = metrics.roc_auc_score(y_test, yPredictedProbabilities)
print (AUC)

# If you randomly chose one positive and one negative observation:
# AUC represents the likelihood that your classifier will assign a
# higher predicted probability to the positive observation
score = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()
print (score)
