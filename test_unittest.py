import unittest
import numpy as np
import HPCRequest as rqs


# test the sequence extraction
class TestSequence(unittest.TestCase):
    def test_failed_init(self):
        with self.assertRaises(AssertionError):
            rqs.Workload([])

    def test_init_default(self):
        wl = rqs.Workload([3])
        self.assertEqual(wl.default_interpolation, True)
        self.assertEqual(len(wl.fit_model), 1)
        self.assertTrue(wl.best_fit is not None)
        wl = rqs.Workload([3]*101)
        self.assertEqual(wl.default_interpolation, True)
        self.assertTrue(wl.fit_model is None)
        self.assertTrue(wl.best_fit is None)

    def test_init_discrete(self):
        wl = rqs.Workload([3, 3, 5], interpolation_model=[])
        self.assertEqual(wl.default_interpolation, False)
        self.assertTrue(wl.fit_model is None)
        self.assertTrue(wl.best_fit is None)

    def test_init_discrete(self):
        wl = rqs.Workload([3, 3, 5],
                          interpolation_model=[rqs.PolyInterpolation()])
        self.assertEqual(wl.default_interpolation, False)
        self.assertEqual(len(wl.fit_model), 1)
        self.assertTrue(wl.best_fit is not None)

    def test_discrete(self):
        wl = rqs.Workload([3, 3, 5, 7, 9 ,9], interpolation_model=[])
        discrete_data, discrete_cdf = wl.compute_cdf()
        cdf = [i / 6 for i in [2, 3, 4, 6]]
        self.assertEqual(discrete_data, [3, 5, 7, 9])
        self.assertEqual(discrete_cdf, cdf)
        wl = rqs.Workload([5]*101)
        discrete_data, cdf = wl.compute_cdf()
        self.assertEqual(discrete_data, [5])
        self.assertAlmostEqual(cdf[0], 1, places=1)


# test the cost model
class TestCostModel(unittest.TestCase):
    def test_with_checkpoint(self):
        sequence = [(4,1), (10, 0)]
        handler = rqs.LogDataCost(sequence)
        self.assertEqual(len(handler.sequence), 2)
        self.assertEqual(handler.sequence[0], 4)

    def test_without_checkpoint(self):
        sequence = [4, 10]
        handler = rqs.LogDataCost(sequence)
        self.assertEqual(len(handler.sequence), 2)
        self.assertEqual(handler.sequence[0], 4)
    
    def test_cost_without_checkpoint(self):
        sequence = [(4, 0), (10, 0)]
        handler = rqs.LogDataCost(sequence)
        self.assertEqual(handler.compute_cost([3]), 3)
        self.assertEqual(handler.compute_cost([6]), 10)
        self.assertEqual(handler.compute_cost([12]), 26)

# test the interpolation model
