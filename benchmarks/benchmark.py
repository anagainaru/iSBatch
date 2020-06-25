import sys
sys.path.append("..")
import pytest
import numpy as np
import iSBatch as rqs


def write_cost_to_file(fct_name, param, *args):
    fname = ":".join(fct_name.split("_")[1:])
    with open("metrics.perf", 'a') as fp:
        fp.write("cost %s %s %s\n" % (
            fname, str(param),
            ' '.join([str(i) for i in args])))


def default_sequence(training, history, interpolation, discretization,
                     stype):
    cl = rqs.ClusterCosts(1, 1, 0)
    training = np.append(training, max(history))
    params = rqs.ResourceParameters()
    params.interpolation_model = interpolation,
    params.resource_discretization = discretization,
    params.CR_strategy = stype
    wl = rqs.ResourceEstimator(training, params=params)
    sequence = wl.compute_request_sequence(cluster_cost=cl)
    cost = wl.compute_sequence_cost(sequence, history, cluster_cost=cl)
    return cost


@pytest.mark.parametrize("len_history, interpolation",
                         [(10, 0), (25, 0), (50, 0), (100, 0), (300, 0),
                          (10, 1), (25, 1), (50, 1), (100, 1), (300, 1)])
def test_training_interpolation(benchmark, len_history, interpolation):
    params = str(len_history) + ":" + str(interpolation)
    if interpolation == 0:
        interpolation = []
    else:
        interpolation = rqs.PolyInterpolation()
    history = np.loadtxt("../examples/logs/neuroscience.in", delimiter=' ')
    interpolation = rqs.PolyInterpolation()
    cost = benchmark(default_sequence, history[:len_history], history,
                     interpolation, 500, 0)
    write_cost_to_file(sys._getframe().f_code.co_name, params, cost)
    assert cost > 0


@pytest.mark.parametrize("discretization, interpolation",
                         [(10, 0), (50, 0), (100, 0), (300, 0), (500, 0),
                          (10, 1), (50, 1), (100, 1), (300, 1), (500, 1)])
def test_discretization_interpolation(benchmark, discretization, interpolation):
    params = str(discretization) + ":" + str(interpolation)
    if interpolation == 0:
        interpolation = []
    else:
        interpolation = rqs.PolyInterpolation(discretization=discretization)
    history = np.loadtxt("../examples/logs/neuroscience.in", delimiter=' ')
    interpolation = rqs.PolyInterpolation()
    cost = benchmark(default_sequence, history[:100], history,
                     interpolation, discretization, 0)
    write_cost_to_file(sys._getframe().f_code.co_name, params, cost)
    assert cost > 0


@pytest.mark.parametrize("len_history, stype, interpolation",
                         [(50, 0, 0), (300, 0, 0), (50, 1, 0),
                          (300, 1, 0), (50, 2, 0), (300, 2, 0),
                          (50, 0, 1), (300, 0, 1), (50, 1, 1),
                          (300, 1, 1), (50, 2, 1), (300, 2, 1)])
def test_training_checkpoint_interpolation(benchmark, stype, len_history, interpolation):
    params = str(len_history) + ":" + str(stype) + ":" + str(interpolation)
    if interpolation == 0:
        interpolation = []
    else:
        interpolation = rqs.PolyInterpolation(discretization=100)
    history = np.loadtxt("../examples/logs/neuroscience.in", delimiter=' ')
    cost = benchmark(default_sequence, history[:len_history], history,
                     interpolation, 100, stype)
    write_cost_to_file(sys._getframe().f_code.co_name, params, cost)
    assert cost > 0
