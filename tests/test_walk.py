import numpy as np

from qcost.alias import alias_tables, prepare_toffolis, prepared_distribution
from qcost.amplify import lcu_block_encoding, oblivious_round, rounds_needed, success_after
from qcost.exact import spectrum, tfim, tfim_open_energy
from qcost.walk import walk_matrix


def test_tfim_exact_energy():
    assert np.isclose(spectrum(tfim(6, 0.7))[0], tfim_open_energy(6, 0.7))


def test_walk_eigenphases_are_arccos_E_over_lambda():
    H = tfim(4, 0.7)
    E = spectrum(H)
    W = walk_matrix(H)
    assert np.allclose(W.conj().T @ W, np.eye(len(W)))
    phases = np.abs(np.angle(np.linalg.eigvals(W)))
    for e in E:
        assert np.min(np.abs(phases - np.arccos(e / H.one_norm))) < 1e-10


def test_alias_sampling_error_within_bound():
    rng = np.random.default_rng(0)
    for L, mu in [(5, 3), (11, 6), (40, 8)]:
        w = rng.random(L)
        keep, alt, n = alias_tables(w, mu)
        p = prepared_distribution(keep, alt, mu)
        assert np.allclose(p, n / (L * 2**mu))
        assert np.abs(p - w / w.sum()).max() <= 1 / (2**mu * L) + 1e-15
    assert prepare_toffolis(16, 10)["uniform superposition"] == 0


def test_amplitude_amplification():
    H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    A = np.kron(np.kron(H1, H1), H1)
    good = np.zeros(8, dtype=bool)
    good[5] = True
    m = rounds_needed(1 / 8)
    assert success_after(A, good, m) > 0.9 > success_after(A, good, 0)


def test_oblivious_amplitude_amplification_one_round():
    phi = 0.4
    c = (2 - np.cos(phi) - np.sin(phi)) / 2
    X, I2 = np.array([[0, 1], [1, 0]]), np.eye(2)
    U = lcu_block_encoding([np.cos(phi) + c, c, np.sin(phi) / 2, np.sin(phi) / 2], [I2, -I2, 1j * X, 1j * X])
    V = np.cos(phi) * I2 + 1j * np.sin(phi) * X
    assert np.allclose(U[:2, :2], V / 2)
    assert np.allclose(oblivious_round(U, 4)[:2, :2], V)
