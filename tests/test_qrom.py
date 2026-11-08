import numpy as np
import pytest

from qcost.circuit import read_int, write_int
from qcost.qrom import best_k, qroam_toffolis, qrom, qrom_toffolis, select_swap_qrom


@pytest.mark.parametrize("L", [1, 2, 3, 5, 8, 11, 16, 37])
@pytest.mark.parametrize("controlled", [False, True])
def test_qrom_outputs_every_word(L, controlled):
    b = 5
    data = np.random.default_rng(L).integers(0, 2**b, L)
    c, addr, out, ctrl = qrom(data, b, controlled)
    reps = 2 if controlled else 1
    X = np.zeros((L * reps, c.n), dtype=bool)
    write_int(X, addr, np.tile(np.arange(L), reps))
    if controlled:
        X[L:, ctrl] = True
    Y = c.run(X)
    on = X[:, ctrl] if controlled else np.ones(len(X), dtype=bool)
    assert (read_int(Y, out) == np.where(on, data[np.tile(np.arange(L), reps)], 0)).all()
    assert c.toffolis() == qrom_toffolis(L, controlled)


@pytest.mark.parametrize("L", [3, 13, 37])
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_select_swap_outputs_every_word(L, k):
    b = 4
    data = np.random.default_rng(L).integers(0, 2**b, L)
    c, addr, out = select_swap_qrom(data, b, k)
    X = np.zeros((L, c.n), dtype=bool)
    write_int(X, addr, np.arange(L))
    assert (read_int(c.run(X), out) == data).all()
    # the circuit is uncontrolled: two fewer than the formula's ceil(L/k) (one for the control, one for rounding)
    assert c.toffolis() == max(0, int(np.ceil(L / k)) - 2) + b * (k - 1)


def test_best_k_near_sqrt_L_over_b():
    k, cost = best_k(lambda k: qroam_toffolis(10_000, 20, k))
    assert k in (16, 32) and cost < 10_000 / 4
