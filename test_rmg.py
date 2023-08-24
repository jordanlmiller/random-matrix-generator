import unittest
import numpy as np

from rmg import RMG, Real, Complex
from util import array_equal, adjoint, inner

class Testrmg(unittest.TestCase):

    def setUp(self):
        """run prior to test"""
        self.n = n = 10
        self.k = 10
        self.base_rmg = RMG(n)
        self.real_rmg = Real(n)
        self.complex_rmg = Complex(n)


    def tearDown(self):
        """run after each test"""
        pass


    def test_permutation(self):
        base_permutation = self.base_rmg.permutation()
        real_permutation = self.real_rmg.permutation()
        complex_permutation = self.complex_rmg.permutation()
        for row in range(self.n):
            self.assertTrue(np.sum(base_permutation[row,:]), 1)
            self.assertTrue(np.sum(real_permutation[row,:]), 1)
            self.assertTrue(np.sum(complex_permutation[row,:]), 1)
        for col in range(self.n):
            self.assertTrue(np.sum(base_permutation[:,col]), 1)
            self.assertTrue(np.sum(real_permutation[:,col]), 1)
            self.assertTrue(np.sum(complex_permutation[col,:]), 1)


    def test_Real_input(self):
        self.assertRaises(TypeError, Real, "string")


    def test_Real_normal(self):
        for _ in range(self.k):
            a = self.real_rmg.normal()
            a_star = adjoint(a)
            self.assertTrue(array_equal(a@a_star, a_star@a))


    def test_Real_orthogonal(self):
        for _ in range(self.k):
            a = self.real_rmg.orthogonal()
            self.assertTrue(array_equal(a@np.transpose(a), np.eye(self.n)))


    def test_Real_special_orthogonal(self):
        for _ in range(self.k):
            a = self.real_rmg.special_orthogonal()
            self.assertTrue(array_equal(a@np.transpose(a), np.eye(self.n)))
            determinant = np.linalg.det(a)
            self.assertAlmostEqual(determinant, 1.)


    def test_Real_symmetric(self):
        for _ in range(self.k):
            a = self.real_rmg.symmetric()
            self.assertTrue(array_equal(a, np.transpose(a)))


    def test_Real_skew_symmetric(self):
        for _ in range(self.k):
            a = self.real_rmg.skew_symmetric()
            self.assertTrue(array_equal(a, -np.transpose(a)))


    def test_Real_positive_definite(self):
            for _ in range(self.k):
                a = self.real_rmg.skew_symmetric()
                self.assertTrue(array_equal(a, -np.transpose(a)))




    def test_Complex_input(self):
        self.assertRaises(TypeError, Complex, "string") 

    
    def test_Complex_normal(self):
        for _ in range(self.k):
            a = self.complex_rmg.normal()
            a_star = adjoint(a)
            self.assertTrue(array_equal(a@a_star, a_star@a))
    

    def test_Complex_unitary(self):
        for _ in range(self.k):
            a = self.complex_rmg.unitary()
            a_star = adjoint(a)
            self.assertTrue(array_equal(a@a_star, np.eye(self.n)))


    def test_Complex_special_unitary(self):
        for _ in range(self.k):
            a = self.complex_rmg.special_unitary()
            a_star = adjoint(a)
            self.assertTrue(array_equal(a@a_star, np.eye(self.n)))
            determinant = np.linalg.det(a)
            det_magnitude = np.sqrt(determinant.real**2 + determinant.imag**2)
            self.assertAlmostEqual(det_magnitude, 1)


    def test_Complex_hermitian(self):
        for _ in range(self.k):
            a = self.complex_rmg.hermitian()
            a_star = adjoint(a)
            self.assertTrue(array_equal(a, a_star))


    def test_Complex_skew_hermitian(self):
        for _ in range(self.k):
            a = self.complex_rmg.skew_hermitian()
            a_star = adjoint(a)
            self.assertTrue(array_equal(a, -a_star))


    def test_Complex_symmetric(self):
        for _ in range(self.k):
            a = self.complex_rmg.symmetric()
            self.assertTrue(array_equal(a, np.transpose(a)))


    def test_Complex_skew_symmetric(self):
        for _ in range(self.k):
            a = self.complex_rmg.skew_symmetric()
            self.assertTrue(array_equal(a, -np.transpose(a)))




if __name__ == '__main__':
    unittest.main()