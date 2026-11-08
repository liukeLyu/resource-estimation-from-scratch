"""PREPARE by coherent alias sampling: load the weights ``|c_l| / lambda`` into an index register.

The classical alias method (Walker 1977; Vose 1991) turns any distribution over ``L`` items into
``L`` equal buckets, each holding at most two items: item ``l`` keeps a share ``keep_l / 2^mu`` of
its own bucket and gives the rest to an alias ``alt_l``. Babbush et al. [arXiv:1805.03662,
Sec. III D, Fig. 11] make it coherent:

1. a uniform superposition over ``l < L`` and over a ``mu``-bit register ``sigma``;
2. a QROM writes ``(keep_l, alt_l)`` next to ``l``: ``L - 1`` Toffolis whatever the word length;
3. a comparator sets a flag when ``sigma >= keep_l``: ``mu`` Toffolis;
4. controlled on the flag, swap ``l`` with ``alt_l``: ``ceil(log2 L)`` Toffolis.

The probabilities come out as ``rho~_l = n_l / (L 2^mu)`` for integers ``n_l``, so each is within
``1 / (2^mu L)`` of the target [same paper, Eq. 35]. The total, ``L + mu + O(log L)`` Toffolis
(``4 (L + mu) + O(log L)`` T in the paper), is set by the QROM; select/swap lowers it.
"""

from __future__ import annotations

import numpy as np

from .qrom import n_bits, qroam_toffolis


def alias_tables(weights, mu: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integer alias tables for ``weights``: returns ``(keep, alt, n)`` with ``sum n = L 2^mu``.

    ``n_l`` is ``L 2^mu |w_l| / sum|w|`` rounded so the counts add up exactly (largest remainder).
    """
    w = np.abs(np.asarray(weights, dtype=float))
    L, B = len(w), 2**mu
    target = w / w.sum() * L * B
    n = np.floor(target).astype(np.int64)
    short = L * B - n.sum()
    n[np.argsort(target - n)[::-1][:short]] += 1
    left = n.copy()
    keep = np.full(L, B, dtype=np.int64)
    alt = np.arange(L)
    small = [l for l in range(L) if left[l] < B]
    large = [l for l in range(L) if left[l] >= B]
    while small and large:
        s, g = small.pop(), large[-1]
        keep[s], alt[s] = left[s], g
        left[g] -= B - left[s]
        if left[g] < B:
            large.pop()
            small.append(g)
    return keep, alt, n


def prepared_distribution(keep, alt, mu: int) -> np.ndarray:
    """What the coherent alias sampler prepares: ``l`` stays if ``sigma < keep_l``, else becomes ``alt_l``."""
    L, B = len(keep), 2**mu
    p = keep / (L * B)
    np.add.at(p, alt, (B - keep) / (L * B))
    return p


def uniform_toffolis(L: int, b_r: int = 7) -> int:
    """Uniform superposition over ``L`` items: free if ``L`` is a power of 2, else ``3 n - 3 eta + 2 b_r - 9``.

    ``n = ceil(log2 L)``, ``2^eta`` the largest power of 2 dividing ``L``, ``b_r`` bits for the one
    rotation that fixes the amplitude [Lee et al., arXiv:2011.03494, App. C, step 1(a)].
    """
    if L & (L - 1) == 0:
        return 0
    eta = (L & -L).bit_length() - 1
    return 3 * n_bits(L) - 3 * eta + 2 * b_r - 9


def prepare_toffolis(L: int, mu: int, k: int = 1) -> dict:
    """Toffolis of alias-sampling PREPARE over ``L`` items with ``mu`` bits, QROM block size ``k``."""
    word = n_bits(L) + mu  # alt_l and keep_l
    parts = {
        "uniform superposition": uniform_toffolis(L),
        "QROM for keep, alt": L - 1 if k == 1 else qroam_toffolis(L, word, k),
        "comparator": mu,
        "controlled swap": n_bits(L),
    }
    parts["total"] = sum(parts.values())
    return parts


def mu_for(lam: float, eps: float) -> int:
    """Keep-register bits for energy error ``eps``: ``ceil(log2(2 sqrt(2) lambda / eps))`` [arXiv:1805.03662, Eq. 36].

    The two further terms of Eq. 36 are ``O(1)`` and dropped here.
    """
    return int(np.ceil(np.log2(2 * np.sqrt(2) * lam / eps)))
