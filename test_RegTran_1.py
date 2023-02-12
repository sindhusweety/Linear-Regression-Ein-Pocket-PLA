# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 13:03:10 2023

@author: jacksonj
"""

from RegTran_1 import reg_tran_1
import numpy as np

def test_reg_tran_1():
    # Load the test sets
    with np.load("RegTran_1_test_sets.npz") as npz_file:

        # Obtain list of arrays
        file_names = npz_file.files
        file_name_iter = iter(file_names)

        # Check proposed solution against reference solution
        # for each test set
        for _ in range(5):
            X = npz_file[next(file_name_iter)]
            y = npz_file[next(file_name_iter)]
            EinClass = reg_tran_1(X, y)
            EinClass_ref = npz_file[next(file_name_iter)]
            assert np.allclose(EinClass, EinClass_ref)
    
if __name__ == "__main__":
    test_reg_tran_1()
    print("Success")