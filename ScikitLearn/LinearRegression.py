from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
import seaborn as sns
import numpy as np

data = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv",index_col = 0)

# Visualize the relationship between the features and the response and show
# the linear regression
sns.pairplot(data,x_vars = ['TV','Radio','Newspaper'],y_vars = 'Sales',size = 7,aspect = 0.7,kind = 'reg')
sns.plt.show()

# Preparing X and y using pandas
feature_cols = ['TV','Radio','Newspaper']
X = data[feature_cols]
y = data['Sales']

# Split data into training and testing sets
# Default split is 75% for training and 25% for testing
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 1)

# Apply linear regression
linreg = LinearRegression()
linreg.fit(X_train,y_train)

# Print the intercept and the coefficients
print (linreg.intercept_)
print (linreg.coef_)

# y = 2.88 + 0.0466 * TV + 0.179 * Radio + 0.00345 * Newspaper

# Pair the feature names with the coefficients
featureList = zip(feature_cols,linreg.coef_)
print (list(featureList))

# Make predictions
yPredictions = linreg.predict(X_test)

# Evaluation metrics
score = metrics.mean_absolute_error(y_test,yPredictions)
print (score)
score = metrics.mean_squared_error(y_test,yPredictions)
print (score)
score = np.sqrt(metrics.mean_squared_error(y_test,yPredictions))
print (score)

# RMSE = 1.40 seems pretty good as our sales ranged from 5 to 25

# Applying feature selection: We notice that the newspaper feature does not
# have a strong correlation. We therefore repeat our calculations without
# using newspaper as a feature

feature_cols = ['TV','Radio']
X = data[feature_cols]
y = data['Sales']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 1)
linreg.fit(X_train,y_train)
yPredictions = linreg.predict(X_test)
score = np.sqrt(metrics.mean_squared_error(y_test,yPredictions))
print (score)

# Notice that our RMSE is lower and thus our new model is performing slightly
# better than the previous model. Therefore the newspaper feature should be
# left out of the model
