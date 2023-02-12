# -*- coding: utf-8 -*-

# TODO: SAVE THIS FILE AS RegTran_2.py

import numpy as np
from scipy import linalg

SEED = 424242 # Random number generator seed value

def reg_tran_2(X, y):
    '''
    Given classification problem on separable data, compare number
    of updates made by PLA when started with all-0 vector and when started
    with weight vector produced by linear regression.

    Parameters
    ----------
    X : N x 2 NumPy array
        N points in d-dimensional space.
    y : N-element NumPy vector
        +1/-1 labels for each of the X points.

    Returns
    -------
    iter_zeros: integer
        Number of weight updates performed when all-0 vector is used as 
        initial weight vector 
    iter_regression: integer
    	Number of weight updates performed when the regression w is used as 
        initial weight vector
    '''
    # Create the return variables

    # TODO: AT LEAST SOME OF YOUR CODE GOES BELOW THIS LINE

    iter_zeros = iter_zerosAlg(X, y)
    iter_regression = iter_regressionAlg(X, y)

    # Print and return the number of iterations made by each algorithm
    print(f"Number of iterations starting with all-0 w:      {iter_zeros}")
    print(f"Number of iterations starting with regression w: {iter_regression}")
    return (iter_zeros, iter_regression)
    
# TODO: YOU MIGHT ALSO WANT TO ADD CODE BELOW THIS LINE


def seed_generator():
    SEED = 424242  # Random number generator seed value
    rng = np.random.default_rng(SEED)
    return rng


def iter_zerosAlg(X, y):
    X = np.insert(X, obj=0, values=1, axis=1)  # axis 0 -> rows 1 -> col & obj -> index
    rng = seed_generator()
    w = np.zeros(X.shape[1])
    iter_zeros = 0
    Ein = np.mean(np.sign(np.dot(X, w.T)) != y)
    while Ein > 0.0:

        predicted_y = np.sign(X @ w)

        misclassified_indices = np.nonzero(predicted_y != y)[0]
        if misclassified_indices.size > 0:
            misclassified_index = rng.choice(misclassified_indices)
            w = w + (y[misclassified_index] * X[misclassified_index])
            Ein = np.mean(np.sign(np.dot(X, w)) != y)
            iter_zeros += 1
    return iter_zeros

def iter_regressionAlg(X, y):
    X = np.insert(X, obj=0, values=1, axis=1)  # axis 0 -> rows 1 -> col & obj -> index
    rng = seed_generator()
    w = linalg.lstsq(X, y)[0]
    iter_regression = 0
    Ein = np.mean(np.sign(np.dot(X, w.T)) != y)
    while Ein > 0.0:

        predicted_y = np.sign(X @ w)

        misclassified_indices = np.nonzero(predicted_y != y)[0]
        if misclassified_indices.size > 0:
            misclassified_index = rng.choice(misclassified_indices)
            w = w + (y[misclassified_index] * X[misclassified_index])
            Ein = np.mean(np.sign(np.dot(X, w)) != y)
            iter_regression += 1
    return iter_regression