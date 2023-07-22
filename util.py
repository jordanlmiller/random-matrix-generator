"""Utility functions for use in rmg.py"""

import numpy as np


def pseudo_inverse(A):
    '''return the moore-penrose pseudo inverse of the given matrix'''
    pass


def adjoint(A):
    '''return the adjoint of the given matrix A'''
    tr = np.transpose(A)
    return np.conjugate(A)


def column_gram_schmidt(A):
    '''perform gram-schmidt orthogonalization on the columns of a matrix'''
    pass