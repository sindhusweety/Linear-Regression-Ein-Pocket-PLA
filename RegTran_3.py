# -*- coding: utf-8 -*-

# TODO: SAVE THIS FILE AS RegTran_3.py

import numpy as np
from scipy import linalg



def seed_generator():
    SEED = 424242  # Random number generator seed value
    rng = np.random.default_rng(SEED)
    return rng

def reg_tran_3(X, y, T):
    '''
    Computes the Ein values for four different classification algorithms,
    some of which use regression and/or nonlinear transformation.
    The training data is not separable.

    Parameters
    ----------
    X : N x 2 NumPy array
        N points in d-dimensional space.
    y : N-element NumPy vector
        +1/-1 labels for each of the X points.
    T : non-negative integer.
        Number of iterations for the Pocket Algorithm.

    Returns
    -------
	Ein_pock: real 
        the Ein for the Pocket algorithm based on a PLA that is 
        initialized with an all-zero weight vector and that randomly 
        selects a misclassified input to use in each weight update
    Ein_pock_tran: real
        the Ein for the Pocket algorithm as above but run on data transformed 
        using the Φ(x) from Problem 1
    Ein_RfC: real
        the Ein for regression used as a classifier by applying sign() 
        to its hypothesis
    Ein_RfC_tran: real 
        the Ein for the algorithm from Problem 1 (regression run on 
        transformed data and then used as a classifier by applying sign() 
        to its hypothesis)

    '''
    # Create the return variables
    # TODO: AT LEAST SOME OF YOUR CODE GOES BELOW THIS LINE
    X_ = np.insert(X, obj=0, values=1, axis=1)  # axis 0 -> rows 1 -> col & obj -> index
    transformed_X = np.array([[1, x[0], x[1], x[0] ** 2, x[0] * x[1], x[1] ** 2] for x in X])
    Ein_pock = Ein_pockAlg(X_, y, T)
    Ein_pock_tran = Ein_pockAlg(transformed_X, y, T)
    Ein_RfC = Ein_linear_regression(X_, y)
    Ein_RfC_tran = Ein_linear_regression(transformed_X, y)
    # Print and return the Ein for each algorithm
    print(f"Ein for pocket:                  {Ein_pock}")
    print(f"Ein for pocket on transform:     {Ein_pock_tran}")
    print(f"Ein for regression:              {Ein_RfC}")
    print(f"Ein for regression on transform: {Ein_RfC_tran}")
    return (Ein_pock, Ein_pock_tran, Ein_RfC, Ein_RfC_tran)

# TODO: YOU MIGHT ALSO WANT TO ADD CODE BELOW THIS LINE

def Ein_pockAlg(X, y, T):

    rng = seed_generator()
    w = np.zeros(X.shape[1])
    min_Ein = np.mean(np.sign(np.dot(X, w.T)) != y)
    for t in range(T):
        predicted_y = np.sign(X @ w)
        misclassified_indices = np.nonzero(predicted_y != y)[0]
        if misclassified_indices.size > 0:
            misclassified_index = rng.choice(misclassified_indices)
            w = w + (y[misclassified_index] * X[misclassified_index])
            Ein = np.mean(np.sign(np.dot(X, w)) != y)

            if Ein < min_Ein:
                min_Ein = Ein
    return min_Ein


def Ein_linear_regression(X, y):

    w = linalg.lstsq(X, y)[0]
    Ein = np.mean(np.sign(np.dot(X, w.T)) != y)
    return Ein


