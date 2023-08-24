"""Random matrix generator class"""

import numpy as np

import util

class RMG:

    def __init__(self, n, seed=None):
        if isinstance(n, int):
            self._n = n
            self._rng = np.random.default_rng(seed)
        else:
            raise TypeError("Dimension must be int. Provided dimension is of class {}".format(type(n)))
    
    def __repr__(self):
        print("Random Matrix Generator for {}-dimensional Square Matrices".format(self._n))

    def _u_variate(self, *params):
        return self._rng.uniform(low=params[0], high=params[1], size=(self._n, self._n))
    
    def permutation(self):
        return np.eye(self._n)[self._rng.permutation(self._n)]
            


class Real(RMG):

    def __repr__(self):
        print("Random Matrix Generator for Real-Valued {}-dimensional Square Matrices".format(self._dimension))

    def naive_random(self):
        return self._u_variate(-1, 1)

    def normal(self):
        r = self.naive_random()
        return r @ np.transpose(r)
    
    def orthogonal(self):
        r = self.naive_random()
        return util.column_gram_schmidt(r)
            
    def special_orthogonal(self):
        singular = True
        while singular:
            ortho = self.orthogonal()
            det = np.linalg.det(ortho)
            if np.sign(det) == 1.:
                singular = False
            elif np.sign(det) == -1.:
                singular = False
                index = self._rng.choice(self._n)
                ortho[:,index] = -ortho[:,index]
        return ortho

    def symmetric(self):
        r = self.naive_random()
        return r + np.transpose(r)

    def skew_symmetric(self):
        r = self.naive_random()
        return r - np.transpose(r)
    
    def positive_definite(self):
        r = self.naive_random()
        return r**2




class Complex(RMG):

    def __repr__(self):
        print("Random Matrix Generator for Complex-Valued {}-dimensional Square Matrices".format(self._dimension))

    def naive_random(self):
        theta = self._u_variate(0, 2*np.pi)
        return np.cos(theta) + 1.j*np.sin(theta)

    def normal(self):
        c = self.naive_random()
        return c @ util.adjoint(c)
    
    def unitary(self):
        c = self.naive_random()
        return util.column_gram_schmidt(c)

    def special_unitary(self):
        singular = True
        while singular:
            u = self.unitary()
            det = np.linalg.det(u)
            magnitude = np.sqrt(det.real**2 + det.imag**2)
            if np.abs(magnitude - 1) < 0.001:
                singular = False
        angle = np.arctan2(det.real, det.imag)
        root_angle = - angle / self._n
        return u / (np.cos(root_angle) + 1.j*np.sin(root_angle))

    def hermitian(self):
        c = self.naive_random()
        return c + util.adjoint(c)

    def skew_hermitian(self):
        c = self.naive_random()
        return c - util.adjoint(c)

    def symmetric(self):
        c = self.naive_random()
        return c + np.transpose(c)

    def skew_symmetric(self):
        c = self.naive_random()
        return c - np.transpose(c)
