import numpy as np
import pytest

from qcost.arithmetic import adder, comparator, phase_gradient_rotation
from qcost.circuit import all_inputs, read_int


@pytest.mark.parametrize("n", range(1, 7))
def test_adder_adds_every_input(n):
    c, a, b = adder(n)
    X = all_inputs([a, b], c.n)
    Y = c.run(X)
    A, B = read_int(X, a), read_int(X, b)
    assert (read_int(Y, b) == (A + B) % 2**n).all()
    assert (read_int(Y, a) == A).all()
    assert not Y[:, 2 * n :].any()  # carries back to 0
    assert c.toffolis() == n - 1


@pytest.mark.parametrize("n", range(1, 7))
def test_comparator_compares_every_input(n):
    c, a, b, flag = comparator(n)
    X = all_inputs([a, b], c.n)
    Y = c.run(X)
    A, B = read_int(X, a), read_int(X, b)
    assert (Y[:, flag] == (A < B)).all()
    assert (read_int(Y, a) == A).all() and (read_int(Y, b) == B).all()
    assert Y.sum(1).tolist() == (X.sum(1) + (A < B)).tolist()  # nothing else left behind
    assert c.toffolis() == n


def test_phase_gradient_gives_the_rotation():
    for b in (3, 5):
        for theta in range(2**b):
            assert np.isclose(phase_gradient_rotation(theta, b), np.exp(2j * np.pi * theta / 2**b))
