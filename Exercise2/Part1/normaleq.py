import numpy as np

def normalEquation(X, y):
    #extract size of row and column
    n,m = X.shape

    #create column of ones
    ones = np.ones((n, 1))

    #append column with ones to the matrix
    newX = np.hstack((X, ones))

    #transpose matrix
    XT = newX.T
    #multiply two matrix
    XTX = XT.dot(newX)
    #find inverse of matrix
    invXTX = np.linalg.inv(XTX)

    XTy = XT.dot(y)

    w = invXTX.dot(XTy)

    print(w)
