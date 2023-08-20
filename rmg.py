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
        a = self._n_variate()
        return util.column_gram_schmidt(a)

    def special_orthogonal(self):
        ortho = self.orthogonal()
        det = np.linalg.det(ortho)
        special_ortho = ortho / (det ** (1/self._n))
        return special_ortho

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
        c = self._n_variate() + 1.j*self._n_variate()
        return c @ util.adjoint(c)
    
    def unitary(self):
        c = self._n_variate() + 1.j*self._n_variate()
        return util.column_gram_schmidt(c)

    def special_unitary(self):
        u = self.unitary()
        det = np.linalg.det(u)
        angle = np.arctan2(det.real, det.imag)
        root_angle = - angle / self._n
        return u / (np.cos(root_angle) + 1.j*np.sin(root_angle))

    def hermitian(self):
        c = self._n_variate() + 1.j*self._n_variate()
        return c + util.adjoint(c)

    def skew_hermitian(self):
        c = self._n_variate() + 1.j*self._n_variate()
        return c - util.adjoint(c)

    def symmetric(self):
        c = self._n_variate() + 1.j*self._n_variate()
        return c + np.transpose(c)

    def skew_symmetric(self):
        c = self._n_variate() + 1.j*self._n_variate()
        return c - np.transpose(c)

    def permutation(self):
        return np.eye(self._n)[self._rng.permutation(self._n)]