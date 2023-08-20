"""Utility functions for use in rmg.py"""

import numpy as np


def array_equal(A, B, eps):
    '''determine if two arrays are equal up to a small value eps'''
    abs_diff = np.abs(A-B)
    return np.all(abs_diff < eps)


def pseudo_inverse(A):
    '''return the moore-penrose pseudo inverse of the given matrix'''
    rows, cols = A.shape
    if cols > rows:
        A_star = adjoint(A)
        return np.linalg.inv(A_star @ A) @ A_star
    elif rows > cols:
        A_star = adjoint(A)
        return A_star @ np.linalg.inv(A @ A_star)
    else:
        return np.linalg.inv(A)  


def adjoint(A):
    '''return the adjoint of the given matrix A'''
    tr = np.transpose(A)
    return np.conjugate(A)


def inner(u, v):
    """return the inner product of the vectors, u and v"""
    return np.dot(u, np.conjugate(v))


def proj(u, v):
    '''return the projection of the vector v onto the vector u'''
    return (inner(v,u) / inner(u,u)) * u


def column_gram_schmidt(A):
    '''perform gram-schmidt orthogonalization on the columns of a matrix'''
    _, cols = A.shape
    for i in range(cols):
        for j in range(i):
            A[:,i] -= proj(A[:,j], A[:,i])
        A[:,i] /= np.sqrt(inner(A[:,i], A[:,i]))

    return A