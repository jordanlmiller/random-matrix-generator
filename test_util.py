import unittest
import numpy as np

import util


class TestUtil(unittest.TestCase):

    def setUp(self):
        """run prior to test"""
        self.rng = np.random.default_rng()

    def tearDown(self):
        """run after each test"""
        pass

    def test_array_equal(self):
        v = self.rng.random(10)+1
        z = np.zeros_like(v)
        self.assertFalse(util.array_equal(v, -v))
        self.assertFalse(util.array_equal(v,z))

    def test_column_gram_schmidt(self):
        a  = self.rng.random((10,10))
        ortho = util.column_gram_schmidt(a)
        rows, cols = ortho.shape
        for i in range(cols):
            for j in range(cols):
                if i == j:
                    self.assertAlmostEqual(util.inner(ortho[:,i], ortho[:,j]), 1)
                elif i != j:
                    self.assertAlmostEqual(util.inner(ortho[:,i], ortho[:,j]), 0)




if __name__ == '__main__':
    unittest.main()