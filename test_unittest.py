import unittest
import numpy as np
import HPCRequest as rqs
from scipy.stats import norm

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

    def test_init_continuous(self):
        wl = rqs.Workload([3, 3, 5],
                          interpolation_model=[rqs.PolyInterpolation()])
        self.assertEqual(wl.default_interpolation, False)
        self.assertEqual(len(wl.fit_model), 1)
        self.assertTrue(wl.best_fit is not None)

    def test_discrete_fit(self):
        wl = rqs.Workload([3, 3, 5, 7, 9 ,9], interpolation_model=[])
        discrete_data, discrete_cdf = wl.compute_cdf()
        cdf = [i / 6 for i in [2, 3, 4, 6]]
        self.assertEqual(discrete_data, [3, 5, 7, 9])
        self.assertEqual(discrete_cdf, cdf)
        wl = rqs.Workload([5]*101)
        discrete_data, cdf = wl.compute_cdf()
        self.assertEqual(discrete_data, [5])
        self.assertAlmostEqual(cdf[0], 1, places=1)

    def test_continuous_fit(self):
        wl = rqs.Workload([5]*10)
        discrete_data, cdf = wl.compute_cdf()
        self.assertEqual(discrete_data, [5])
        self.assertAlmostEqual(cdf[0], 1, places=1)

    def test_compute_sequence(self):
        wl = rqs.Workload([5]*10)
        sequence = wl.compute_request_sequence()
        self.assertEqual(sequence, [(5,)])
        wl = rqs.Workload([5]*101)
        sequence = wl.compute_request_sequence()
        self.assertEqual(sequence, [(5,)])

    def test_example_sequences(self):
        # test the default model (alpha 1, beta 0, gamma 0)
        history = np.loadtxt("log_examples/truncnorm.in", delimiter=' ')
        wl = rqs.Workload(history,
                          interpolation_model=[rqs.DistInterpolation(
                              history, list_of_distr=[norm])])
        sequence = wl.compute_request_sequence()
        self.assertTrue(abs(sequence[0][0] - 10.8) < 0.1)
        wl = rqs.Workload(history)
        sequence = wl.compute_request_sequence()
        self.assertTrue(abs(sequence[0][0] - 10.8) < 0.1)

        history = np.loadtxt("log_examples/neuroscience.in", delimiter=' ')
        wl = rqs.Workload(history)
        sequence = wl.compute_request_sequence()
        self.assertTrue(abs(sequence[0][0]/3600 - 22.4) < 0.1)

    def test_configurations(self):
        # test the HPC model (alpha 1 beta 1 gamma 0)
        history = np.loadtxt("log_examples/truncnorm.in", delimiter=' ')
        wl = rqs.Workload(history)
        sequence = wl.compute_request_sequence(alpha=1, beta=1, gamma=0)
        self.assertTrue(abs(sequence[0][0] - 11.2) < 0.1)
        history = np.loadtxt("log_examples/neuroscience.in", delimiter=' ')
        wl = rqs.Workload(history)
        sequence = wl.compute_request_sequence(alpha=1, beta=1, gamma=0)
        self.assertTrue(abs(sequence[0][0]/3600 - 23.8) < 0.1)

# test the cost model
class TestCostModel(unittest.TestCase):
    def test_cost_with_checkpoint(self):
        sequence = [(4,1), (10, 0)]
        handler = rqs.LogDataCost(sequence)
        self.assertEqual(len(handler.sequence), 2)
        self.assertEqual(handler.sequence[0], 4)

    def test_cost_without_checkpoint(self):
        sequence = [4, 10]
        handler = rqs.LogDataCost(sequence)
        self.assertEqual(len(handler.sequence), 2)
        self.assertEqual(handler.sequence[0], 4)
        self.assertEqual(handler.compute_cost([3]), 3)
        self.assertEqual(handler.compute_cost([6]), 10)
        self.assertEqual(handler.compute_cost([12]), 26)

    def test_sequence_cost(self):
        wl = rqs.Workload([5]*101)
        sequence = wl.compute_request_sequence()
        cost = wl.compute_sequence_cost(sequence, [1, 2, 3])
        self.assertEqual(cost, 2)
