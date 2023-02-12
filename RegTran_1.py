# -*- coding: utf-8 -*-

# TODO: SAVE THIS FILE AS RegTran_1.py

import numpy as np
from scipy import linalg

def reg_tran_1(X, y):
    '''
    Given classification problem data, use nonlinear transformation and 
    linear regression to produce a classifier.

    Parameters
    ----------
    X : N x 2 NumPy array
        N points in d-dimensional space.
    y : N-element NumPy vector
        +1/-1 labels for each of the X points.

    Returns
    -------
    EinClass : (d+1)-element NumPy vector
        EinClass, fraction of training points misclassified by the 
        transformation+regression algorithm. 
    '''
    # Create the return variable

    # TODO: AT LEAST SOME OF YOUR CODE GOES BELOW THIS LINE
    #Nonlinearly transform the X data using the feature transform
    # Φ(𝐱) = (1, 𝑥1 , 𝑥2 , 𝑥12 , 𝑥1 𝑥2 , 𝑥22 )
    feature_tranform = np.array( [[1,x[0], x[1], x[0]**2, x[0] * x[1], x[1]**2]   for x in X])
    # Print and return EinClass
    w = linalg.lstsq(feature_tranform, y)[0]
    misclassified = np.sum(np.sign(feature_tranform @ w) != y)
    EinClass = misclassified / y.shape[0]
    print(f"EinClass = {EinClass}")

    return EinClass

# TODO: YOU MIGHT ALSO WANT TO ADD CODE BELOW THIS LINE