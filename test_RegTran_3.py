# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 20:24:13 2022

@author: jacksonj
"""

from RegTran_3 import reg_tran_3
import numpy as np


def test_reg_tran_3():

    # Load the test sets
    with np.load("RegTran_3_test_sets.npz") as npz_file:

        # Obtain list of arrays
        file_names = npz_file.files
        file_name_iter = iter(file_names)

        # Check proposed solution against reference solution
        # for each test set
        for _ in range(5):
            X = npz_file[next(file_name_iter)]
            y = npz_file[next(file_name_iter)]
            T = npz_file[next(file_name_iter)]
            results = reg_tran_3(X, y, T)
            results_ref = npz_file[next(file_name_iter)]
            print(results_ref)
            assert np.allclose(results, results_ref)

if __name__ == "__main__":
    test_reg_tran_3()
    print("Success")
