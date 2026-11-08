"""Amplitude amplification, plain and oblivious, on small state vectors.

Plain [Brassard, Hoyer, Mosca, Tapp, quant-ph/0005055, Thm. 2]: if a circuit ``A`` succeeds with
probability ``a = sin^2(theta)``, then ``m`` rounds of ``Q = -A S_0 A^dag S_good`` raise the success
amplitude to ``sin((2m + 1) theta)``; ``m = floor(pi / (4 theta))`` rounds suffice, about
``1 / sqrt(a)`` instead of the ``1 / a`` repetitions of trying again.

Oblivious [Berry et al., arXiv:1312.1414, Lemma 3.6]: if ``U |0>|psi> = sin(theta) |0> V|psi> + ...``
for a unitary ``V`` and every ``|psi>``, the same rotation works without knowing ``|psi>``, using
only reflections about the ancilla ``|0>``. When ``sin(theta) = 1/2``, one round,
``-U R U^dag R U`` with ``R = 2|0><0| - 1`` on the ancilla, gives ``V|psi>`` exactly.
"""

from __future__ import annotations

import numpy as np


def grover_iterate(A: np.ndarray, good: np.ndarray) -> np.ndarray:
    """``Q = -A S_0 A^dag S_good``: ``S_good`` flips the sign of good basis states, ``S_0`` of ``|0>``."""
    dim = A.shape[0]
    S_good = np.diag(np.where(good, -1.0, 1.0))
    S_0 = np.eye(dim)
    S_0[0, 0] = -1
    return -A @ S_0 @ A.conj().T @ S_good


def success_after(A: np.ndarray, good: np.ndarray, m: int) -> float:
    """Probability of a good outcome after ``m`` rounds of amplitude amplification."""
    psi = A[:, 0]
    Q = grover_iterate(A, good)
    for _ in range(m):
        psi = Q @ psi
    return float(np.sum(np.abs(psi[good]) ** 2))


def rounds_needed(a: float) -> int:
    """``floor(pi / (4 theta))`` with ``sin^2(theta) = a`` [quant-ph/0005055, Thm. 2]."""
    return int(np.floor(np.pi / (4 * np.arcsin(np.sqrt(a)))))


def lcu_block_encoding(coeffs, unitaries) -> np.ndarray:
    """``U = (PREP^dag x 1) SELECT (PREP x 1)`` for ``sum_j c_j U_j``, ``c_j > 0``; ancilla is the first factor.

    The top-left block of ``U`` is ``sum_j c_j U_j / lambda``. Needs a power of 2 number of terms.
    """
    c = np.asarray(coeffs, dtype=float)
    k, d = len(c), unitaries[0].shape[0]
    first = np.sqrt(c / c.sum())
    # any unitary with this first column: a Householder reflection
    e0 = np.zeros(k)
    e0[0] = 1
    v = first - e0
    prep = np.eye(k) - 2 * np.outer(v, v) / (v @ v) if v @ v > 1e-15 else np.eye(k)
    select = np.zeros((k * d, k * d), dtype=complex)
    for j, Uj in enumerate(unitaries):
        select[j * d : (j + 1) * d, j * d : (j + 1) * d] = Uj
    P = np.kron(prep, np.eye(d))
    return P.conj().T @ select @ P


def oblivious_round(U: np.ndarray, n_anc_dim: int) -> np.ndarray:
    """One round of oblivious amplitude amplification, ``-U R U^dag R U``, with ``R = 2|0><0| - 1`` on the ancilla."""
    d = U.shape[0] // n_anc_dim
    R = -np.eye(U.shape[0])
    R[:d, :d] += 2 * np.eye(d)
    return -U @ R @ U.conj().T @ R @ U
