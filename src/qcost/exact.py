"""Hamiltonians as Pauli sums, and exact references to check against.

``PauliSum`` holds ``H = sum_l c_l P_l`` as a list of ``(c_l, "XZIY...")`` with qubit 0 the
leftmost letter. This is the form the block encoding uses: ``L`` terms and the one-norm
``lambda = sum_l |c_l|``.

The transverse-field Ising chain ``H = -sum X_i X_{i+1} - h sum Z_i`` is the running example.
The Jordan-Wigner transformation makes it a quadratic fermion Hamiltonian (Lieb, Schultz,
Mattis 1961; Pfeuty 1970). For the open chain its single-particle energies are ``2 s_n``, where
``s_n`` are the singular values of the ``n x n`` bidiagonal matrix with ``M_ii = h`` and
``M_{i,i+1} = J``, and the ground energy is ``E_0 = -sum_n s_n``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, identity, kron

PAULI = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]),
}


@dataclass
class PauliSum:
    """``H = sum_l c_l P_l`` with real ``c_l`` and Pauli words ``P_l`` on ``n`` qubits."""

    terms: list[tuple[float, str]]

    @property
    def n(self) -> int:
        return len(self.terms[0][1])

    @property
    def L(self) -> int:
        return len(self.terms)

    @property
    def one_norm(self) -> float:
        return float(sum(abs(c) for c, _ in self.terms))

    def matrix(self):
        """Sparse ``2^n x 2^n`` matrix of ``H``."""
        dim = 2**self.n
        H = csr_matrix((dim, dim), dtype=complex)
        for c, word in self.terms:
            H = H + c * pauli_matrix(word)
        return H


def pauli_matrix(word: str):
    """Sparse matrix of one Pauli word, qubit 0 leftmost."""
    M = identity(1, format="csr", dtype=complex)
    for ch in word:
        M = kron(M, csr_matrix(PAULI[ch]), format="csr")
    return M


def tfim(n: int, h: float = 1.0, J: float = 1.0) -> PauliSum:
    """Open TFIM chain ``-J sum X_i X_{i+1} - h sum Z_i``: ``2n - 1`` terms, ``lambda = (n - 1) J + n h``."""
    terms = []
    for i in range(n - 1):
        terms.append((-J, "I" * i + "XX" + "I" * (n - i - 2)))
    for i in range(n):
        terms.append((-h, "I" * i + "Z" + "I" * (n - i - 1)))
    return PauliSum(terms)


def tfim_open_energy(n: int, h: float = 1.0, J: float = 1.0) -> float:
    """Exact ground-state energy of the open chain of ``n`` sites, from free fermions."""
    M = h * np.eye(n) + J * np.eye(n, k=1)
    return -float(np.sum(np.linalg.svd(M, compute_uv=False)))


def spectrum(H: PauliSum) -> np.ndarray:
    """All eigenvalues of a small ``H`` by dense diagonalization."""
    return np.linalg.eigvalsh(H.matrix().toarray())


def from_openfermion(op, n: int) -> PauliSum:
    """``PauliSum`` from an OpenFermion ``QubitOperator`` on ``n`` qubits (the identity term is dropped)."""
    terms = []
    for term, c in op.terms.items():
        if not term:
            continue
        word = ["I"] * n
        for q, p in term:
            word[q] = p
        terms.append((float(np.real(c)), "".join(word)))
    return PauliSum(terms)
