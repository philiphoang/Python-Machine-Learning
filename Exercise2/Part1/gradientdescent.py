import numpy as np

def step_gradient(w0_current, w1_current, points, learningRate):
    #initialize the partial derivatives for the cummulative sum
    w0_par_der = 0
    w1_par_der = 0

    n = len(points)

    #computation for the summation
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # partial derivative (of MSE) with respect to w0
        w0_par_der += (y - ((w1_current * x) + w0_current))
        # partial derivative (of MSE) with respect to w1
        w1_par_der += x * (y - ((w1_current * x) + w0_current))

    #multiplication of summation results with -2/n
    w0_par_der = -(2/n) * w0_par_der
    w1_par_der = -(2/n) * w1_par_der

    #make a gradient vector from the partial derivatives
    gradient_mse = np.array([w0_par_der, w1_par_der])

    #make a vector of weights
    weight_vector = np.array([w0_current, w1_current])

    #update rule for weights
    updated_weight_vector = weight_vector - (learningRate * gradient_mse)

    #return the updated weight vector as a list
    return np.ndarray.tolist(updated_weight_vector)
