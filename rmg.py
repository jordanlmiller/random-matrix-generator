"""Random matrix generator class"""

import numpy as np

import util

class RMG:

    def __init__(self, n, seed=None):
        self._n = n
        self._rng = np.random.default_rng(seed)
    
    def __repr__(self):
        print("Random Matrix Generator for {}-dimensional Square Matrices".format(self._n))

    def _u_variate(self):
        return self._rng.uniform((self._n, self._n))
    
    def _n_variate(self):
        return self._rng.normal(size=(self._n, self._n))
            


class Real(RMG):

    def __repr__(self):
        print("Random Matrix Generator for Real-Valued {}-dimensional Square Matrices".format(self._dimension))

    def normal(self):
        a = self._n_variate()
        return a @ np.transpose(a)
    
    def orthogonal(self):
        pass

    def special_orthogonal(self):
        pass

    def symmetric(self):
        a = self._n_variate()
        return a + np.transpose(a)

    def skew_symmetric(self):
        a = self._n_variate()
        return a - np.transpose(a)

    def permutation(self):
        return np.eye(self._n)[self._rng.permutation(self._n)]



class Complex(RMG):

    def __repr__(self):
        print("Random Matrix Generator for Complex-Valued {}-dimensional Square Matrices".format(self._dimension))

    def normal(self):
        a = self._uniform()
        return a @ util.adjoint(a)
    
    def unitary(self):
        pass

    def special_unitary(self):
        pass

    def hermitian(self):
        pass

    def skew_hermitian(self):
        pass

    def symmetric(self):
        a = self._n_variate()
        return a + np.transpose(a)

    def skew_symmetric(self):
        a = self._n_variate()
        return a - np.transpose(a)


    def permutation(self):
        return np.eye(self._n)[self._rng.permutation(self._n)]