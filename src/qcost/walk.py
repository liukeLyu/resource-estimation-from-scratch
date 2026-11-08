"""The qubitized walk for a Pauli sum: build it as a matrix, check its spectrum, count its Toffolis.

Write ``H = sum_l c_l P_l`` with ``L`` terms and ``lambda = sum_l |c_l|``. PREPARE puts
``sum_l sqrt(|c_l| / lambda) |l>`` in an index register; SELECT applies ``sign(c_l) P_l`` controlled
on ``|l>``. Then ``(PREP^dag x 1) SELECT (PREP x 1)`` has ``H / lambda`` in its top-left block, and
the walk ``W = R PREP^dag SELECT PREP``, with ``R`` the reflection about the index register's
``|0>``, has eigenvalues ``exp(+- i arccos(E_k / lambda))`` for every eigenvalue ``E_k`` of ``H``
[Low and Chuang, arXiv:1610.06546; Babbush et al., arXiv:1805.03662, Sec. II].

One walk step costs one SELECT, one PREPARE and its inverse, and one reflection:

* SELECT: unary iteration over ``L`` items, ``L - 1`` Toffolis; the controlled Paulis are Clifford.
* PREPARE: alias sampling (``alias.prepare_toffolis``), twice.
* reflection about zero on the ``ceil(log2 L)`` index and ``mu`` superposition qubits, controlled
  by one more qubit for phase estimation: a NOT with ``n`` controls costs ``n - 1`` Toffolis
  [Gidney, arXiv:1709.06648, p. 5], so ``ceil(log2 L) + mu`` here.

Phase estimation then needs ``ceil(pi lambda / (2 eps))`` walk steps for energy error ``eps``
[Lee et al., arXiv:2011.03494, Eq. 45]. Babbush et al. define ``eps`` as a root-mean-square
error instead and need ``sqrt(2) pi lambda / eps`` [arXiv:1805.03662, Eq. 26].
"""

from __future__ import annotations

import numpy as np

from .alias import mu_for, prepare_toffolis
from .exact import PauliSum, pauli_matrix
from .qrom import n_bits


def walk_matrix(H: PauliSum) -> np.ndarray:
    """Dense walk operator ``R (PREP^dag SELECT PREP)`` with the index register as the first factor."""
    L, n = H.L, H.n
    m = n_bits(L)
    K, d = 2**m, 2**n
    first = np.zeros(K)
    first[:L] = np.sqrt(np.abs([c for c, _ in H.terms]) / H.one_norm)
    e0 = np.zeros(K)
    e0[0] = 1
    v = first - e0
    prep = np.eye(K) - 2 * np.outer(v, v) / (v @ v) if v @ v > 1e-15 else np.eye(K)
    select = np.zeros((K * d, K * d), dtype=complex)
    for l in range(K):
        if l < L:
            c, word = H.terms[l]
            block = np.sign(c) * pauli_matrix(word).toarray()
        else:
            block = np.eye(d)  # unused indices: identity keeps SELECT self-inverse
        select[l * d : (l + 1) * d, l * d : (l + 1) * d] = block
    P = np.kron(prep, np.eye(d))
    U = P.conj().T @ select @ P
    R = -np.eye(K * d)
    R[:d, :d] += 2 * np.eye(d)
    return R @ U


def walk_toffolis(H: PauliSum, eps: float, k: int = 1) -> dict:
    """Toffolis of one walk step for ``H`` at energy error ``eps``, with a breakdown."""
    L = H.L
    mu = mu_for(H.one_norm, eps)
    prep = prepare_toffolis(L, mu, k)["total"]
    parts = {"SELECT": L - 1, "PREPARE": prep, "PREPARE inverse": prep, "reflection": n_bits(L) + mu}
    parts["total"] = sum(parts.values())
    parts["mu"] = mu
    return parts


def walk_calls(lam: float, eps: float) -> int:
    """Walk steps for phase estimation to error ``eps``: ``ceil(pi lambda / (2 eps))`` [arXiv:2011.03494, Eq. 45]."""
    return int(np.ceil(np.pi * lam / (2 * eps)))
