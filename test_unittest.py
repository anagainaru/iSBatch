import unittest
import numpy as np
import iSBatch as rqs
from scipy.stats import norm
import warnings

def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test


class TestEstimationParameters(unittest.TestCase):
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
        params = rqs.ResourceParameters()
        params.interpolation_model=[]
        wl = rqs.ResourceEstimator([3, 3, 5], params=params)
        self.assertEqual(wl.default_interpolation, False)
        self.assertTrue(wl.fit_model is None)
        self.assertTrue(wl.best_fit is None)

    def test_init_continuous(self):
        params = rqs.ResourceParameters()
        params.interpolation_model=[rqs.PolyInterpolation()]
        wl = rqs.ResourceEstimator([3, 3, 5], params=params)
        self.assertEqual(wl.default_interpolation, False)
        self.assertEqual(len(wl.fit_model), 1)	
        wl.compute_request_sequence()
        self.assertTrue(wl.best_fit is not None)

    def test_discrete_fit(self):
        params = rqs.ResourceParameters()
        params.interpolation_model=[]
        wl = rqs.ResourceEstimator([3, 3, 5, 7, 9 ,9],
                                   params=params)
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

    def test_default_discretization(self):
        with self.assertRaises(AssertionError):
            params = rqs.ResourceParameters()
            params.resource_discretization=2
            rqs.ResourceEstimator([i for i in range(10)], params=params)
        wl = rqs.ResourceEstimator([i for i in range(10)])
        self.assertEqual(wl.discretization, 500)
        # the default sequence for 10 walltime history uses interpolation
        data, cdf = wl._get_cdf()
        self.assertEqual(len(data), 500)
        wl = rqs.ResourceEstimator([i for i in range(101)])
        data, cdf = wl._get_cdf()
        self.assertEqual(len(data), 101)

    def test_custom_discretization(self):
        params = rqs.ResourceParameters()
        params.resource_discretization=100
        wl = rqs.ResourceEstimator([i for i in range(10)],
                                   params=params)
        self.assertEqual(wl.discretization, 100)
        # the default sequence for 10 walltime history uses interpolation
        data, cdf = wl._get_cdf()
        self.assertEqual(len(data), 100)
        params = rqs.ResourceParameters()
        params.resource_discretization=50
        wl = rqs.ResourceEstimator([i for i in range(101)],
                                   params=params)
        data, cdf = wl._get_cdf()
        self.assertEqual(len(data), 50)
        params = rqs.ResourceParameters()
        params.resource_discretization=200
        wl = rqs.ResourceEstimator([i for i in range(101)],
                                   params=params)
        data, cdf = wl._get_cdf()
        self.assertEqual(len(data), 200)

    @ignore_warnings
    def test_reservation_limits(self):
        history = np.loadtxt("examples/logs/truncnorm.in", delimiter=' ')
        params = rqs.ResourceParameters()
        params.request_upper_limit = 12.5
        wl = rqs.ResourceEstimator(history, params=params)
        sequence = wl.compute_request_sequence()
        self.assertTrue(all([i[0] <= 12.5 for i in sequence]))
        params = rqs.ResourceParameters()
        params.request_lower_limit = 12
        wl = rqs.ResourceEstimator(history, params=params)
        sequence = wl.compute_request_sequence()
        self.assertTrue(all([i[0] >= 12 for i in sequence]))
        params = rqs.ResourceParameters()
        params.request_upper_limit = 12.5
        params.request_lower_limit = 12
        wl = rqs.ResourceEstimator(history, params=params)
        sequence = wl.compute_request_sequence()
        self.assertTrue(all([i[0] >= 12 and i[0] <= 12.5 for i in sequence]))

    def test_reservation_limits_interpolation(self):
        history = np.loadtxt("examples/logs/truncnorm.in", delimiter=' ')
        params = rqs.ResourceParameters()
        params.interpolation_model = rqs.PolyInterpolation()
        params.resource_discretization = 2400
        params.request_upper_limit = 12.5
        params.request_lower_limit=12
        wl = rqs.ResourceEstimator(history, params=params)
        sequence = wl.compute_request_sequence()
        self.assertTrue(all([i[0] >= 12 and i[0] <= 12.5 for i in sequence]))

    def test_increment_limit(self):
        history = np.loadtxt("examples/logs/CT_eye_segmentation.log",
                             delimiter=' ')
        params = rqs.ResourceParameters()
        params.request_increment_limit = 1800
        params.CR_strategy = rqs.CRStrategy.NeverCheckpoint
        wl = rqs.ResourceEstimator(history)
        sequence = wl.compute_request_sequence()
        self.assertTrue(sequence[0][0] >= 1800)
        self.assertTrue(all(sequence[i][0] - sequence[i-1][0] >= 1800
                            for i in range(1, len(sequence))))
        params = rqs.ResourceParameters()
        params.request_increment_limit = 1800
        params.CR_strategy = rqs.CRStrategy.AlwaysCheckpoint
        wl = rqs.ResourceEstimator(history)
        sequence = wl.compute_request_sequence()
        # since it's all checkpoint every reservation represents the increment
        self.assertTrue(all(i[0] >= 1800 for i in sequence))

# test the sequence extraction
class TestSequence(unittest.TestCase):
    def test_failed_init(self):
        with self.assertRaises(AssertionError):
            rqs.ResourceEstimator([])

    @ignore_warnings
    def test_compute_sequence(self):
        wl = rqs.ResourceEstimator([5]*10)
        sequence = wl.compute_request_sequence()
        self.assertEqual(sequence, [(5, 0)])
        wl = rqs.ResourceEstimator([5]*101)
        sequence = wl.compute_request_sequence()
        self.assertEqual(sequence, [(5, 0)])

    @ignore_warnings
    def test_example_sequence_checkpoint(self):
        history = np.loadtxt("examples/logs/truncnorm.in", delimiter=' ')
        history = history[:10]
        params = rqs.ResourceParameters()
        params.CR_strategy = rqs.CRStrategy.AdaptiveCheckpoint
        params.interpolation_model = []
        wl = rqs.ResourceEstimator(history, params=params)
        sequence = wl.compute_request_sequence()
        self.assertEqual(len(sequence), 3)
        self.assertEqual(sequence[0][1], 1)
        self.assertTrue(np.sum([i[0] for i in sequence]) >= max(history))
        time_adapt = np.sum([i[0] for i in sequence if i[1]==1])
        time_adapt += sequence[len(sequence)-1][0]
        params = rqs.ResourceParameters()
        params.CR_strategy = rqs.CRStrategy.AlwaysCheckpoint
        params.interpolation_model = []
        wl = rqs.ResourceEstimator(history, params=params)
        sequence = wl.compute_request_sequence()
        self.assertTrue(np.sum([i[0] for i in sequence]) >= max(history))
        # check that the total execution covered is the same in both
        time = np.sum([i[0] for i in sequence if i[1]==1])
        time += sequence[len(sequence)-1][0]
        self.assertTrue(time == time_adapt)

    def test_example_sequences(self):
        # test the default model (alpha 1, beta 1, gamma 0)
        history = np.loadtxt("examples/logs/truncnorm.in", delimiter=' ')
        params = rqs.ResourceParameters()
        params.interpolation_model=[rqs.DistInterpolation(
                              list_of_distr=[norm],
                              discretization=len(history))]
        wl = rqs.ResourceEstimator(history, params=params)
        sequence = wl.compute_request_sequence()
        self.assertTrue(abs(sequence[0][0] - 11.5) < 0.1)
        params = rqs.ResourceParameters()
        params.interpolation_model=[rqs.DistInterpolation(
                              list_of_distr=[norm],
                              discretization=100)]
        wl = rqs.ResourceEstimator(history, params=params)
        sequence = wl.compute_request_sequence()
        self.assertTrue(abs(sequence[0][0] - 11.5) < 0.1)
        wl = rqs.ResourceEstimator(history)
        sequence = wl.compute_request_sequence()
        self.assertTrue(abs(sequence[0][0] - 11.2) < 0.1)

        history = np.loadtxt("examples/logs/CT_eye_segmentation.log", delimiter=' ')
        wl = rqs.ResourceEstimator(history)
        sequence = wl.compute_request_sequence()
        self.assertTrue(abs(sequence[0][0]/3600 - 23.8) < 0.1)

    def test_system_models(self):
        # test the Cloud model (alpha 1 beta 0 gamma 0)
        history = np.loadtxt("examples/logs/truncnorm.in", delimiter=' ')
        wl = rqs.ResourceEstimator(history)
        sequence = wl.compute_request_sequence(cluster_cost=rqs.ClusterCosts(
            reservation_cost = 1, utilization_cost=0, deploy_cost=0))
        self.assertTrue(abs(sequence[0][0] - 10.8) < 0.1)
        params = rqs.ResourceParameters()
        params.interpolation_model=[rqs.DistInterpolation(
                              list_of_distr=[norm],
                              discretization=100)]
        wl = rqs.ResourceEstimator(history, params=params)
        sequence = wl.compute_request_sequence(cluster_cost=rqs.ClusterCosts(
            reservation_cost = 1, utilization_cost=0, deploy_cost=0))
        self.assertTrue(abs(sequence[0][0] - 10.8) < 0.1)
        history = np.loadtxt("examples/logs/CT_eye_segmentation.log", delimiter=' ')
        wl = rqs.ResourceEstimator(history)
        sequence = wl.compute_request_sequence(cluster_cost=rqs.ClusterCosts(
            reservation_cost = 1, utilization_cost=0, deploy_cost=0))
        self.assertTrue(abs(sequence[0][0]/3600 - 22.4) < 0.1)


# test the sequence extraction
class TestLimitedSequence(unittest.TestCase):
    def test_failed_init(self):
        params = rqs.ResourceParameters()
        params.submissions_limit = 0
        wl = rqs.ResourceEstimator([i for i in range(1000)],
                                    params=params)
        with self.assertRaises(AssertionError):
            sequence = wl.compute_request_sequence()

    def get_average_submissions(self, sequence, history):
        submissions = 0
        for i in history:
            compute = 0
            for s in sequence:
                if i > s[0] + compute:
                    submissions += 1
                # if the application was checkpointed
                if s[1] == 1:
                    compute += s[0]
            # add the successful run
            submissions += 1
        return submissions / len(history)

    def limited_submission(self, limit, strategy):
        history = np.loadtxt('examples/logs/CT_eye_segmentation.log',
                             delimiter=' ')
        params = rqs.ResourceParameters()
        params.submissions_limit = limit
        params.submissions_limit_strategy = strategy
        params.CR_strategy = rqs.CRStrategy.NeverCheckpoint
        wl = rqs.ResourceEstimator(history, params=params)
        sequence = wl.compute_request_sequence()
        submissions1 = len(sequence)
        if strategy == rqs.LimitStrategy.AverageBased:
            submissions1 = self.get_average_submissions(sequence, history)
        params.CR_strategy = rqs.CRStrategy.AlwaysCheckpoint
        wl = rqs.ResourceEstimator(history, params=params)
        sequence = wl.compute_request_sequence()
        submissions2 = len(sequence)
        if strategy == rqs.LimitStrategy.AverageBased:
            submissions2 = self.get_average_submissions(sequence, history)
        params.CR_strategy = rqs.CRStrategy.AdaptiveCheckpoint
        return [submissions1, submissions2]

    @ignore_warnings
    def test_thredhold_limit(self):
        sequence_lens = self.limited_submission(
                1, rqs.LimitStrategy.ThresholdBased)
        self.assertTrue(all(n <= 1 for n in sequence_lens))
        sequence_lens = self.limited_submission(
                2, rqs.LimitStrategy.ThresholdBased)
        self.assertTrue(all(n <= 2 for n in sequence_lens))

    @ignore_warnings
    def test_average_limit(self):
        sequence_lens = self.limited_submission(
                1.5, rqs.LimitStrategy.AverageBased)
        self.assertTrue(all(n <= 2 for n in sequence_lens))


# test the cost model
class TestCostModel(unittest.TestCase):
    def test_cost_with_checkpoint(self):
        sequence = [(4, 1), (6, 0)]
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

    @ignore_warnings
    def test_sequence_cost(self):
        wl = rqs.ResourceEstimator([5]*101)
        sequence = wl.compute_request_sequence()
        cost = wl.compute_sequence_cost(sequence, [1, 2, 3])
        self.assertEqual(cost[0], 7)
        cost = rqs.ClusterCosts(0, 1, 0)
        sequence = wl.compute_request_sequence(cluster_cost=cost)
        cost = wl.compute_sequence_cost(sequence, [1, 2, 3],
                                        cluster_cost=cost)
        self.assertEqual(cost[0], 2)

    def test_cost_validity(self):
        data = np.loadtxt("./examples/logs/truncnorm.in", delimiter=' ')
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
