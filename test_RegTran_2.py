# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 13:03:10 2023

@author: jacksonj
"""

from RegTran_2 import reg_tran_2
import numpy as np

def test_reg_tran_2():
    # Load the test sets
    with np.load("RegTran_2_test_sets.npz") as npz_file:

        # Obtain list of arrays
        file_names = npz_file.files
        file_name_iter = iter(file_names)

        # Check proposed solution against reference solution
        # for each test set
        for _ in range(5):
            X = npz_file[next(file_name_iter)]
            y = npz_file[next(file_name_iter)]
            (iter_zeros, iter_regression) = reg_tran_2(X, y)
            (iter_zeros_ref, 
             iter_regression_ref) = npz_file[next(file_name_iter)]

            assert np.allclose((iter_zeros, iter_regression), 
                               (iter_zeros_ref, iter_regression_ref))
    
if __name__ == "__main__":
    test_reg_tran_2()
    print("Success")