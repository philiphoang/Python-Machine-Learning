import pandas as pd
from sklearn.linear_model import LinearRegression

#make a dataframe of the dataframe
df = pd.read_csv('movies.csv')

#show first five rows of df
df.head(n=5)

#Extract the first column and set it to the output or dependent variable y
y = df[['revenue']]

#Remove the first column and set the rest of the dataframe to X
#This is the set of independent variables
X = df.drop(columns=['revenue'])

#Show first five rows of X
X.head(n=5)

#show first five rows of y
y.head(n=5)

print(X)
print(y)

#Make a lin_reg object from the LinearRegression class
lin_reg = LinearRegression()

#Use the fit method of LinearRegression class to fit a straight line through the data
lin_reg.fit(X, y)

#Display the learned parameters
lin_reg.intercept_, lin_reg.coef_
