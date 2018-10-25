import numpy as np
#plot data
import matplotlib.pyplot as plt
from gradientrunner import gradient_descent_runner
from gradientearlystop import gradient_descent_runner_early_stop
from normaleq import normalEquation
from sklearn.linear_model import LinearRegression

#Exercise 1.1
np.random.seed(2)

#generate 100 x values from 0 to 2 randomly, then sort them in ascending order
X = 2 * np.random.rand(100, 1)
X.sort(axis = 0)

#generate y values and add noise to it
y = 4 + 3 * X + np.random.rand(100, 1)

#matplotlib inline
plt.scatter(X, y)

points = np.column_stack((X, y))

num_iterations = 100
learning_rate = 0.47
initial_w0 = 0 #initial y-intercept guess
initial_w1 = 0 #initial slope guess
threshold = 0.1
[w0, w1, mse] = gradient_descent_runner(points, initial_w0, initial_w1, learning_rate, num_iterations)
#------------

#Exercise 1.2
#[w0, w1, mse] = gradient_descent_runner_early_stop(points, initial_w0, initial_w1, learning_rate, num_iterations, threshold)
#------------

#Exercise 1.3
normalEquation(X,y)

#make a lin_reg object from the LinearRegressionclass
lin_reg = LinearRegression()

#use the fit method of LinearRegression class to fit a straight line through the data
lin_reg.fit(X, y)
print('y-intercept w0:', lin_reg.intercept_)
print('slope w1', lin_reg.coef_)

#plot the original data points as a scatter plot
plt.scatter(X, y, label='original data')

#plot the line that fits these points. Use the values of m and b as provided by the fit method
y_ = lin_reg.coef_*X + lin_reg.intercept_

#you can also get y_ by using the predict method.
#y_ = lin_reg.predict(X)

plt.plot(X, y_, color='r', label='predicted fit')
plt.xlabel('x'); plt.ylabel('y')
plt.legend(loc='best')

plt.show()
#-------------
