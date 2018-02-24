from sklearn.cross_validation import train_test_split
import pandas as pd
import seaborn as sns

'''
# Read CSV file directly from a URL and save the results
data = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv")

# Display the first 5 rows
rows = data.head()
print (rows)
'''

# index_col allows us to set a specific column as the index
# Notice that the first column is now no longer shown as it is being referenced for indices
data = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv",index_col = 0)

# Display the first 5 rows
rows = data.head()
print (rows)

# Display the last 5 rows
rows = data.tail()
print (rows)

# Show the shape of the DataFrame (rows,columns)
shape = data.shape
print (shape)

'''
Features:
TV: Advertising dollars spent on TV
Radio: Advertising dollars spent on Radio
Newspaper: Advertising dollars spent on Newspaper
Values are in thousands of dollars
Response:
Sales: Sales of a single product in thousands of items
'''

# Visualize the relationship between the features and the response
sns.pairplot(data,x_vars = ['TV','Radio','Newspaper'],y_vars = 'Sales',size = 7,aspect = 0.7)
sns.plt.show()

# We can get it to show the linear regression as follows:
sns.pairplot(data,x_vars = ['TV','Radio','Newspaper'],y_vars = 'Sales',size = 7,aspect = 0.7,kind = 'reg')
sns.plt.show()

# Notice that seaborn has added a 95% confidence band

# Preparing X and y using pandas
feature_cols = ['TV','Radio','Newspaper']
X = data[feature_cols]
y = data['Sales']

# Split data into training and testing sets
# Default split is 75% for training and 25% for testing
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 1)
