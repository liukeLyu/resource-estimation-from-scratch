"""The Hubbard model by qubitization, as costed by Babbush et al. [arXiv:1805.03662, Sec. V].

``H = -t sum_<pq>,s a^dag_ps a_qs + u sum_p n_p,up n_p,down`` on an ``M x M`` periodic square
lattice, ``N = 2 M^2`` spin orbitals. Its Pauli form has only three distinct coefficients, so:

* ``lambda = 2 N t + N u / 2`` (Eq. 60; this counts the identity term, ``N u / 8``).
* SELECT: two selected-Majorana unary iterations over ``N`` items and one indexed ``Z`` over
  ``N / 2``: ``4N + 4N + 2N = 10 N`` T gates (Sec. V A, Fig. 19), i.e. ``2.5 N`` Toffolis; translation
  invariance means no data has to be loaded.
* PREPARE: two rotations and uniform superpositions, ``O(log(N / eps))``; no QROM, no alias
  sampling (Sec. V B, Fig. 20). The paper neglects it next to SELECT.
* walk calls ``sqrt(2) pi lambda / Delta E`` for a root-mean-square error ``Delta E`` (Eq. 26).
* ancillas ``12 + 3 log N`` (Eq. 64), with ``ceil(log2 N)`` matching every row of Table IV.
"""

from __future__ import annotations

import numpy as np

# Table IV of arXiv:1805.03662: u/t = 4, Delta E = t/100, periodic, spinful
TABLE_IV = {
    "6 x 6": {"N": 72, "ancilla": 33, "logical": 105, "T": 9.3e7},
    "8 x 8": {"N": 128, "ancilla": 33, "logical": 161, "T": 2.9e8},
    "10 x 10": {"N": 200, "ancilla": 36, "logical": 236, "T": 7.1e8},
    "20 x 20": {"N": 800, "ancilla": 42, "logical": 842, "T": 1.2e10},
}


def hubbard_lambda(N: int, t: float = 1.0, u: float = 4.0) -> float:
    return 2 * N * t + N * u / 2


def hubbard_estimate(N: int, t: float = 1.0, u: float = 4.0, dE: float = 0.01) -> dict:
    """Babbush et al.'s count for ``N`` spin orbitals at RMS error ``dE`` (same units as ``t``)."""
    lam = hubbard_lambda(N, t, u)
    calls = np.sqrt(2) * np.pi * lam / dE
    select_T = 10 * N
    ancilla = 12 + 3 * int(np.ceil(np.log2(N)))
    return {
        "lambda": lam,
        "walk calls": calls,
        "T per call": select_T,
        "T": calls * select_T,
        "Toffolis": calls * select_T / 4,
        "ancilla": ancilla,
        "logical": N + ancilla,
    }


def hubbard_select_toffolis(N: int) -> int:
    """SELECT built from this package's unary iteration: ``(N - 1) + (N - 1) + (N/2 - 1)`` controlled Toffolis."""
    return 2 * (N - 1) + (N // 2 - 1)
