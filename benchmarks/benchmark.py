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
    wl = rqs.ResourceEstimator(training, interpolation_model=interpolation,
                               resource_discretization=discretization,
                               CR_strategy=stype)
    sequence = wl.compute_request_sequence(cluster_cost=cl)
    cost = wl.compute_sequence_cost(sequence, history, cluster_cost=cl)
    return cost


@pytest.mark.parametrize("len_history, interpolation",
                         [(10, 0), (50, 0), (100, 0), (500, 0), (1000, 0),
                          (10, 1), (50, 1), (100, 1), (500, 1), (1000, 1)])
def test_training_interpolation(benchmark, len_history, interpolation):
    params = str(len_history) + ":" + str(interpolation)
    if interpolation == 0:
        interpolation = []
    else:
        interpolation = rqs.PolyInterpolation()
    history = np.loadtxt("../log_examples/truncnorm.in", delimiter=' ')
    interpolation = rqs.PolyInterpolation()
    cost = benchmark(default_sequence, history[:len_history], history,
                     interpolation, 500, 0)
    write_cost_to_file(sys._getframe().f_code.co_name, params, cost)
    assert cost > 0


@pytest.mark.parametrize("discretization, interpolation",
                         [(10, 0), (250, 0), (500, 0), (750, 0), (1000, 0),
                          (10, 1), (250, 1), (500, 1), (750, 1), (1000, 1)])
def test_discretization_interpolation(benchmark, discretization, interpolation):
    params = str(discretization) + ":" + str(interpolation)
    if interpolation == 0:
        interpolation = []
    else:
        interpolation = rqs.PolyInterpolation(discretization=discretization)
    history = np.loadtxt("../log_examples/truncnorm.in", delimiter=' ')
    interpolation = rqs.PolyInterpolation()
    cost = benchmark(default_sequence, history[:100], history,
                     interpolation, discretization, 0)
    write_cost_to_file(sys._getframe().f_code.co_name, params, cost)
    assert cost > 0


@pytest.mark.parametrize("len_history, stype, interpolation",
                         [(100, 0, 0), (1000, 0, 0), (100, 1, 0),
                          (1000, 1, 0), (100, 2, 0), (1000, 2, 0),
                          (100, 0, 1), (1000, 0, 1), (100, 1, 1),
                          (1000, 1, 1), (100, 2, 1), (1000, 2, 1)])
def test_training_checkpoint_interpolation(benchmark, stype, len_history, interpolation):
    params = str(len_history) + ":" + str(stype) + ":" + str(interpolation)
    if interpolation == 0:
        interpolation = []
    else:
        interpolation = rqs.PolyInterpolation(discretization=100)
    history = np.loadtxt("../log_examples/truncnorm.in", delimiter=' ')
    cost = benchmark(default_sequence, history[:len_history], history,
                     interpolation, 100, stype)
    write_cost_to_file(sys._getframe().f_code.co_name, params, cost)
    assert cost > 0
