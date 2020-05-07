import sys
sys.path.append("..")
import iSBatch as rqs
import numpy as np
import pytest

def default_sequence(history, interpolation):
    wl = rqs.ResourceEstimator(history, interpolation_model=interpolation)
    sequence = wl.compute_request_sequence()
    # You may return anything you want, like the result of a computation
    return len(sequence)

@pytest.mark.parametrize("len_history", [(10), (100), (1000)])
def test_history_length_default(benchmark, len_history):
    history = np.loadtxt("../log_examples/truncnorm.in", delimiter=' ')
    history = history[:len_history]
    # benchmark the default sequence for different history lengths
    result = benchmark(default_sequence, history, None)
    assert result > 0

@pytest.mark.parametrize("len_history", [(10), (100), (1000)])
def test_history_length_discrete(benchmark, len_history):
    history = np.loadtxt("../log_examples/truncnorm.in", delimiter=' ')
    history = history[:len_history]
    # benchmark the default sequence for different history lengths
    result = benchmark(default_sequence, history, [])
    assert result > 0


