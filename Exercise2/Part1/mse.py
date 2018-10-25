import numpy as np

#Function which takes the parameters (b and m) of a line and then
#finds the mean-squared error between the user-specified points and line
def compute_error_for_line_given_points(w0, w1, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        #accumulate 'sum of square' errors in totalError variable
        totalError += (y - (w1 * x + w0)) ** 2
    #find mean of sum of squared errors
    mse = totalError/len(points)
    return mse
