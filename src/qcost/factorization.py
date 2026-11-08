"""Double factorization: a molecular Hamiltonian with a smaller one-norm and less data to load.

The two-electron integrals ``V_pqrs = (pq|rs)`` (chemists' order, spatial orbitals) form a positive
semidefinite ``n^2 x n^2`` matrix. Two layers of eigendecompositions [von Burg et al.,
arXiv:2007.14460; Lee et al., arXiv:2011.03494, App. C]:

1. ``V_(pq),(rs) = sum_l W^l_pq W^l_rs``: the first factorization, rank ``L``.
2. each ``W^l = sum_m f^l_m u^l_m (u^l_m)^T``: a one-body operator, diagonal in its own orbitals.

The Hamiltonian becomes a sum of squares of one-body operators, each a rotation of orbitals
followed by number operators. Small pieces are dropped: eigenvector ``m`` of ``W^l`` is kept if
``(sum_p |f^l_p|) |f^l_m| > threshold``, and the first ``l`` with nothing kept ends the list
[Lee et al., Eq. C41]. ``Xi_l`` eigenvectors survive for each ``l``; ``L Xi`` is their total.

The one-norm the block encoding pays [Lee et al., Eq. C11]:
``lambda = sum |eig T'| + (1/4) sum_l (sum_m |f^l_m|)^2``, with the one-body part
``T'_pq = h_pq - (1/2) sum_r (pr|rq) + sum_r (rr|pq)``.

``df_walk_toffolis`` follows the step list of Lee et al., App. C (steps 1 to 10), term by term,
and ``df_walk_toffolis_c39`` their closed form, Eq. C39. OpenFermion's ``resource_estimates``
implements the same step list; ``tests/test_factorization.py`` checks both against it.
"""

from __future__ import annotations

from math import ceil, log2
from pathlib import Path

import numpy as np

from .qrom import best_k


def double_factorize(eri: np.ndarray, thresh: float) -> tuple[list[tuple[np.ndarray, np.ndarray]], int, int]:
    """Factorize ``eri[p, q, r, s] = (pq|rs)``; returns ``(factors, L, L_Xi)``.

    ``factors[l] = (f, U)``: kept eigenvalues ``f`` of ``W^l`` and their eigenvectors as columns of ``U``.
    """
    n = eri.shape[0]
    e, v = np.linalg.eigh(eri.reshape(n * n, n * n))
    order = np.argsort(e)[::-1]
    e, v = e[order], v[:, order]
    keep = e > 0
    W = (v[:, keep] * np.sqrt(e[keep])).reshape(n, n, -1)
    factors, n_vec = [], 0
    for l in range(W.shape[2]):
        f, U = np.linalg.eigh(W[:, :, l])
        sel = np.sum(np.abs(f)) * np.abs(f) > thresh
        if not sel.any():
            break
        factors.append((f[sel], U[:, sel]))
        n_vec += int(sel.sum())
    return factors, len(factors), n_vec


def rebuild_eri(factors) -> np.ndarray:
    """``(pq|rs)`` rebuilt from the kept factors, to see what truncation threw away."""
    Ws = [U @ np.diag(f) @ U.T for f, U in factors]
    return np.einsum("pql,rsl->pqrs", np.stack(Ws, -1), np.stack(Ws, -1), optimize=True)


def one_body_prime(h1: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """``T'_pq = h_pq - (1/2) sum_r (pr|rq) + sum_r (rr|pq)``: the one-body part left after squaring."""
    return h1 - 0.5 * np.einsum("prrq->pq", eri) + np.einsum("rrpq->pq", eri)


def df_lambda(h1: np.ndarray, eri: np.ndarray, factors) -> dict:
    """One-norm of the double-factorized Hamiltonian [Lee et al., Eq. C11]."""
    lam_T = float(np.sum(np.abs(np.linalg.eigvalsh(one_body_prime(h1, eri)))))
    lam_F = float(sum(0.25 * np.sum(np.abs(f)) ** 2 for f, _ in factors))
    return {"one-body": lam_T, "two-body": lam_F, "total": lam_T + lam_F}


def pauli_lambda(h1: np.ndarray, eri: np.ndarray) -> float:
    """One-norm of the Jordan-Wigner Pauli sum of the same Hamiltonian (identity term excluded)."""
    return pauli_sum(h1, eri).one_norm


# Toffoli cost of one walk step [Lee et al., arXiv:2011.03494, App. C]


def _qrom(d, b):
    """Select/swap QROM over ``d`` words of ``b`` bits, best ``k``."""
    return best_k(lambda k: ceil(d / k) + b * (k - 1))[1]


def _unqrom(d):
    """Measurement-based uncomputation of a QROM over ``d`` words, best ``k``."""
    return best_k(lambda k: ceil(d / k) + k)[1]


def _registers(N, L, LXi, aleph, b_r):
    n_L = ceil(log2(L + 1))  # first index l, plus one value for the one-body term
    n_Xi = ceil(log2(N // 2))  # second index m (at most N/2 eigenvectors per l)
    n_LXi = ceil(log2(LXi + N // 2))  # contiguous register over every (l, m)
    eta = ((L + 1) & -(L + 1)).bit_length() - 1  # 2^eta divides L + 1
    return n_L, n_Xi, n_LXi, eta


def df_walk_toffolis(
    N: int, L: int, LXi: int, aleph: int = 10, beth: int = 16, b_r: int = 7, b_r_first: int | None = None
) -> dict:
    """Toffolis of one walk step, step by step [Lee et al., App. C].

    ``N`` spin orbitals, ``L`` first-factorization rank, ``LXi`` eigenvectors kept, ``aleph`` bits
    per alias-sampling keep value, ``beth`` bits per Givens rotation angle, ``b_r`` bits for the
    rotation in an equal superposition. OpenFermion tunes the bits of the first one,
    ``b_r_first``, by minimizing the expected cost of its amplitude amplification: 7 for the
    Reiher FeMoco Hamiltonian, 6 for the Li one.
    """
    b_r_first = b_r if b_r_first is None else b_r_first
    n_L, n_Xi, n_LXi, eta = _registers(N, L, LXi, aleph, b_r)
    b_o = n_Xi + n_LXi + b_r + 1  # data to start the second preparation
    s = {}
    s["1 prepare l (uniform, x2)"] = 2 * (3 * n_L + 2 * b_r_first - 3 * eta - 9)
    s["1 prepare l (QROM + inverse)"] = _qrom(L + 1, n_L + aleph) + _unqrom(L + 1)
    s["1 prepare l (compare, swap, x2)"] = 2 * (aleph + n_L)
    s["2 data for second preparation"] = _qrom(L + 1, b_o) + _unqrom(L + 1)
    s["3 prepare m (uniform, x4)"] = 4 * (7 * n_Xi + 2 * b_r - 6)
    s["3 prepare m (offsets, x4)"] = 4 * (n_LXi - 1)
    b_p2 = n_Xi + aleph + 2
    s["3 prepare m (QROMs + inverses)"] = (
        _qrom(LXi + N // 2, b_p2) + _unqrom(LXi + N // 2) + _qrom(LXi, b_p2) + _unqrom(LXi)
    )
    s["3 prepare m (compare, swap, x4)"] = 4 * (n_Xi + aleph)
    s["4 rotations (offsets)"] = 4 * (n_LXi - 1)
    s["4 rotations (angle QROMs + inverses)"] = (
        _qrom(LXi + N // 2, N * beth // 2) + _unqrom(LXi + N // 2) + _qrom(LXi, N * beth // 2) + _unqrom(LXi)
    )
    s["4 rotations (spin swaps)"] = 2 * N
    s["4 rotations (Givens, phase gradient)"] = 4 * N * (beth - 2)
    s["4 controlled Z"] = 3
    s["6 reflection on m"] = n_Xi + aleph + 2
    s["9 reflection for the walk"] = n_L + n_Xi + aleph + 1
    s["10 control qubits"] = 2
    s["total"] = sum(s.values())
    return s


def df_walk_toffolis_c39(N: int, L: int, LXi: int, aleph: int = 10, beth: int = 16, b_r: int = 7) -> int:
    """Lee et al.'s closed form, Eq. C39, with ``aleph_1 = aleph_2 = aleph`` and each block size optimized."""
    n_L, n_Xi, n_LXi, eta = _registers(N, L, LXi, aleph, b_r)
    b_p1, b_o, b_p2 = n_L + aleph, n_Xi + n_LXi + b_r + 1, n_Xi + aleph + 2
    d1, d2 = LXi + N / 2, LXi
    return int(
        9 * n_L - 6 * eta + 12 * b_r
        + _qrom(L + 1, b_p1) + _qrom(L + 1, b_o) + 2 * _unqrom(L + 1)
        + best_k(lambda k: ceil(d1 / k) + ceil(d2 / k) + N * beth * k)[1]
        + best_k(lambda k: ceil(d1 / k) + ceil(d2 / k) + 2 * k)[1]
        + 34 * n_Xi + 8 * n_LXi
        + best_k(lambda k: ceil(d1 / k) + ceil(d2 / k) + 2 * b_p2 * (k - 1))[1]
        + best_k(lambda k: ceil(d1 / k) + ceil(d2 / k) + 2 * k)[1]
        + 3 * aleph + 6 * aleph + 3 * N * beth - 6 * N - 43
    )


def df_logical_qubits(N: int, L: int, LXi: int, calls: int, aleph: int = 10, beth: int = 16, b_r: int = 7) -> dict:
    """Logical qubits, register by register [Lee et al., App. C, the list after Eq. C39]."""
    n_L, n_Xi, n_LXi, eta = _registers(N, L, LXi, aleph, b_r)
    b_o = n_Xi + n_LXi + b_r + 1
    k_r = best_k(lambda k: ceil((LXi + N // 2) / k) + (N * beth // 2) * (k - 1))[0]
    q = {
        "phase estimation control": 2 * ceil(log2(calls + 1)) - 1,
        "system (spin orbitals)": N,
        "first index l, rotated qubit, flag": n_L + 2,
        "QROM output, superposition, flag": n_L + 2 * aleph + 1,
        "data for second preparation": b_o,
        "second index m, rotated qubit, flag": n_Xi + 2,
        "second QROM output": n_Xi + aleph + 2,
        "second superposition, flag": aleph + 1,
        "rotation angles (k_r words)": k_r * N * beth // 2,
        "phase gradient": beth,
        "spin control, T state": 2,
    }
    q["total"] = sum(q.values())
    return q


# Molecules


def h_chain(n_atoms: int, spacing: float = 1.4) -> tuple[np.ndarray, np.ndarray]:
    """``(h1, eri)`` in the Hartree-Fock orbitals of a linear H chain, STO-3G, ``spacing`` in bohr (needs PySCF)."""
    from pyscf import ao2mo, gto, scf

    atoms = [("H", (0, 0, i * spacing)) for i in range(n_atoms)]
    mol = gto.M(atom=atoms, basis="sto-3g", unit="bohr", verbose=0)
    mf = scf.RHF(mol).run()
    C = mf.mo_coeff
    return C.T @ mf.get_hcore() @ C, ao2mo.restore(1, ao2mo.full(mol, C), C.shape[1])


def pauli_sum(h1: np.ndarray, eri: np.ndarray):
    """The same Hamiltonian as a Jordan-Wigner ``PauliSum`` on ``2 n`` qubits (needs OpenFermion)."""
    from openfermion import InteractionOperator, jordan_wigner
    from openfermion.chem.molecular_data import spinorb_from_spatial

    from .exact import from_openfermion

    one, two = spinorb_from_spatial(h1, eri.transpose(0, 2, 3, 1))
    return from_openfermion(jordan_wigner(InteractionOperator(0.0, one, 0.5 * two)), 2 * h1.shape[0])


def reiher_femoco() -> tuple[np.ndarray, np.ndarray, float]:
    """``(h1, eri, ecore)`` of the 54-orbital FeMoco active space of Reiher et al. [arXiv:1605.03590].

    The file ships with OpenFermion (``resource_estimates/integrals/eri_reiher.h5``, 68 MB).
    """
    import h5py
    import openfermion

    path = Path(openfermion.__file__).parent / "resource_estimates" / "integrals" / "eri_reiher.h5"
    with h5py.File(path, "r") as f:
        return np.asarray(f["h0"]), np.asarray(f["eri"]), float(f["ecore"][()])
