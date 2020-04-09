import unittest
import numpy as np
import iSBatch as rqs
from scipy.stats import norm

# test the sequence extraction
class TestSequence(unittest.TestCase):
    def test_failed_init(self):
        with self.assertRaises(AssertionError):
            rqs.ResourceEstimator([])

    def test_init_default(self):
        wl = rqs.ResourceEstimator([3, 4])
        self.assertEqual(wl.default_interpolation, True)
        wl.compute_request_sequence()
        self.assertEqual(len(wl.fit_model), 1)
        self.assertTrue(wl.best_fit is not None)
        wl = rqs.ResourceEstimator([3]*100 + [4])
        self.assertEqual(wl.default_interpolation, True)
        self.assertTrue(wl.fit_model is None)
        self.assertTrue(wl.best_fit is None)

    def test_init_discrete(self):
        wl = rqs.ResourceEstimator([3, 3, 5], interpolation_model=[])
        self.assertEqual(wl.default_interpolation, False)
        self.assertTrue(wl.fit_model is None)
        self.assertTrue(wl.best_fit is None)

    def test_init_continuous(self):
        wl = rqs.ResourceEstimator([3, 3, 5],
                          interpolation_model=[rqs.PolyInterpolation()])
        self.assertEqual(wl.default_interpolation, False)
        self.assertEqual(len(wl.fit_model), 1)	
        wl.compute_request_sequence()
        self.assertTrue(wl.best_fit is not None)

    def test_discrete_fit(self):
        wl = rqs.ResourceEstimator([3, 3, 5, 7, 9 ,9],
				   interpolation_model=[])
        wl._compute_cdf()
        cdf = [i / 6 for i in [2, 3, 4, 6]]
        self.assertEqual(wl.discrete_data, [3, 5, 7, 9])
        self.assertEqual(wl.cdf, cdf)
        wl = rqs.ResourceEstimator([5]*101)
        wl._compute_cdf()
        self.assertEqual(wl.discrete_data, [5])
        self.assertAlmostEqual(wl.cdf[0], 1, places=1)
        wl = rqs.ResourceEstimator([5]*100 + [6]*100)
        wl._compute_cdf()
        self.assertAlmostEqual(wl.cdf[0], 0.5, places=1)

    def test_continuous_fit(self):
        wl = rqs.ResourceEstimator([5]*10)
        wl._compute_cdf()
        self.assertEqual(wl.discrete_data, [5])
        self.assertAlmostEqual(wl.cdf[0], 1, places=1)
        wl = rqs.ResourceEstimator([5]*10 + [7]*10)
        seq = wl.compute_request_sequence()
        self.assertEqual(seq, [(7, 0)])

    def test_compute_sequence(self):
        wl = rqs.ResourceEstimator([5]*10)
        sequence = wl.compute_request_sequence()
        self.assertEqual(sequence, [(5, 0)])
        wl = rqs.ResourceEstimator([5]*101)
        sequence = wl.compute_request_sequence()
        self.assertEqual(sequence, [(5, 0)])

    def test_example_sequence_checkpoint(self):
        history = np.loadtxt("log_examples/small.in", delimiter=' ')
        wl = rqs.ResourceEstimator(
                history, CR_strategy=rqs.CRStrategy.AdaptiveCheckpoint,
                interpolation_model=[])
        sequence = wl.compute_request_sequence()
        self.assertEqual(len(sequence), 3)
        self.assertEqual(sequence[0][1], 1)
        self.assertTrue(np.sum([i[0] for i in sequence]) >= max(history))

    def test_example_sequences(self):
        # test the default model (alpha 1, beta 1, gamma 0)
        history = np.loadtxt("log_examples/truncnorm.in", delimiter=' ')
        wl = rqs.ResourceEstimator(history,
                          interpolation_model=[rqs.DistInterpolation(
                              list_of_distr=[norm],
                              discretization=len(history))])
        sequence = wl.compute_request_sequence()
        self.assertTrue(abs(sequence[0][0] - 11.5) < 0.1)
        wl = rqs.ResourceEstimator(history,
                          interpolation_model=[rqs.DistInterpolation(
                              list_of_distr=[norm],
                              discretization=100)])
        sequence = wl.compute_request_sequence()
        self.assertTrue(abs(sequence[0][0] - 11.5) < 0.1)
        wl = rqs.ResourceEstimator(history)
        sequence = wl.compute_request_sequence()
        self.assertTrue(abs(sequence[0][0] - 11.2) < 0.1)

        history = np.loadtxt("log_examples/neuroscience.in", delimiter=' ')
        wl = rqs.ResourceEstimator(history)
        sequence = wl.compute_request_sequence()
        self.assertTrue(abs(sequence[0][0]/3600 - 23.8) < 0.1)

    def test_configurations(self):
        # test the Cloud model (alpha 1 beta 0 gamma 0)
        history = np.loadtxt("log_examples/truncnorm.in", delimiter=' ')
        wl = rqs.ResourceEstimator(history)
        sequence = wl.compute_request_sequence(cluster_cost=rqs.ClusterCosts(
            reservation_cost = 1, utilization_cost=0, deploy_cost=0))
        self.assertTrue(abs(sequence[0][0] - 10.8) < 0.1)
        wl = rqs.ResourceEstimator(history,
                          interpolation_model=[rqs.DistInterpolation(
                              list_of_distr=[norm],
                              discretization=100)])
        sequence = wl.compute_request_sequence(cluster_cost=rqs.ClusterCosts(
            reservation_cost = 1, utilization_cost=0, deploy_cost=0))
        self.assertTrue(abs(sequence[0][0] - 10.8) < 0.1)
        history = np.loadtxt("log_examples/neuroscience.in", delimiter=' ')
        wl = rqs.ResourceEstimator(history)
        sequence = wl.compute_request_sequence(cluster_cost=rqs.ClusterCosts(
            reservation_cost = 1, utilization_cost=0, deploy_cost=0))
        self.assertTrue(abs(sequence[0][0]/3600 - 22.4) < 0.1)

# test the cost model
class TestCostModel(unittest.TestCase):
    def test_cost_with_checkpoint(self):
        sequence = [(4, 1), (10, 0)]
        handler = rqs.LogDataCost(sequence)
        cost = rqs.ClusterCosts(1, 0, 0)
        self.assertEqual(handler.compute_cost([3], cost), 4)
        self.assertEqual(handler.compute_cost([7], cost), 10)
        cost = rqs.ClusterCosts(1, 1, 0)
        self.assertEqual(handler.compute_cost([3], cost), 7)
        self.assertEqual(handler.compute_cost([7], cost), 17)
        cost = rqs.ClusterCosts(1, 1, 1)
        self.assertEqual(handler.compute_cost([3], cost), 8)
        self.assertEqual(handler.compute_cost([7], cost), 19)

    def test_cost_without_checkpoint(self):
        sequence = [4, 10]
        handler = rqs.LogDataCost(sequence)
        cost = rqs.ClusterCosts(1, 0, 0)
        self.assertEqual(handler.compute_cost([3], cost), 4)
        self.assertEqual(handler.compute_cost([7], cost), 14)
        cost = rqs.ClusterCosts(1, 1, 0)
        self.assertEqual(handler.compute_cost([3], cost), 7)
        self.assertEqual(handler.compute_cost([7], cost), 25)
        cost = rqs.ClusterCosts(1, 1, 1)
        self.assertEqual(handler.compute_cost([3], cost), 8)
        self.assertEqual(handler.compute_cost([7], cost), 27)

    def test_sequence_cost(self):
        wl = rqs.ResourceEstimator([5]*101)
        sequence = wl.compute_request_sequence()
        cost = wl.compute_sequence_cost(sequence, [1, 2, 3])
        self.assertEqual(cost, 7)
        cost = rqs.ClusterCosts(0, 1, 0)
        sequence = wl.compute_request_sequence(cluster_cost=cost)
        cost = wl.compute_sequence_cost(sequence, [1, 2, 3],
                                        cluster_cost=cost)
        self.assertEqual(cost, 2)

    def test_cost_validity(self):
        data = np.loadtxt("./log_examples/truncnorm.in", delimiter=' ')
        # compute the requests based on the entire data
        wl = rqs.ResourceEstimator(data)
        sequence = wl.compute_request_sequence()
        cost_opt = wl.compute_sequence_cost(sequence, data)
        # compute requests based on part of the data
        wl = rqs.ResourceEstimator(list(data[:10]) + [max(data)])
        sequence = wl.compute_request_sequence()
        cost = wl.compute_sequence_cost(sequence, data)
        self.assertTrue(cost >= cost_opt)
        wl = rqs.ResourceEstimator(list(data[:100]) + [max(data)])
        sequence = wl.compute_request_sequence()
        cost = wl.compute_sequence_cost(sequence, data)
        self.assertTrue(cost >= cost_opt) 
