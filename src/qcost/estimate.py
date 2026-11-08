"""One call: a Hamiltonian in, the cost of its ground-state energy out, every factor itemized.

``estimate(H, eps)`` for a Pauli sum: qubitized phase estimation with alias-sampling PREPARE and
unary-iteration SELECT (``walk.py``), then the surface code (``surface_code.py``).
``estimate_df(...)`` does the same for a molecule in double-factorized form (``factorization.py``).
Both return a dict; ``show`` prints it as a table.
"""

from __future__ import annotations

import numpy as np

from .exact import PauliSum
from .factorization import df_logical_qubits, df_walk_toffolis
from .qrom import n_bits
from .surface_code import footprint
from .walk import walk_calls, walk_toffolis


def estimate(H: PauliSum, eps: float = 0.01, p: float = 1e-3) -> dict:
    """Cost of the ground-state energy of ``H`` to ``eps`` (energy units of ``H``) by qubitized QPE."""
    step = walk_toffolis(H, eps)
    calls = walk_calls(H.one_norm, eps)
    mu, m = step["mu"], n_bits(H.L)
    qubits = {
        "system": H.n,
        "index l": m,
        "alias registers (sigma, keep, alt)": 2 * mu + m,
        "flag and unary-iteration ancillas": 1 + m,
        "phase estimation": int(np.ceil(np.log2(calls + 1))),
    }
    logical = sum(qubits.values())
    toffolis = calls * step["total"]
    return {
        "terms L": H.L,
        "lambda": H.one_norm,
        "keep bits mu": mu,
        "Toffolis per walk step": step["total"],
        "walk calls": calls,
        "Toffolis": toffolis,
        "logical qubits": logical,
        **{f"  step: {k}": v for k, v in step.items() if k not in ("total", "mu")},
        **{f"  qubits: {k}": v for k, v in qubits.items()},
        **footprint(logical, toffolis, p=p),
    }


def estimate_df(N: int, lam: float, L: int, LXi: int, eps: float = 0.0016, beth: int = 16, p: float = 1e-3) -> dict:
    """Same for a double-factorized molecule with ``N`` spin orbitals [Lee et al., arXiv:2011.03494, App. C]."""
    step = df_walk_toffolis(N, L, LXi, beth=beth)["total"]
    calls = walk_calls(lam, eps)
    logical = df_logical_qubits(N, L, LXi, calls, beth=beth)["total"]
    toffolis = calls * step
    return {
        "lambda": lam,
        "Toffolis per walk step": step,
        "walk calls": calls,
        "Toffolis": toffolis,
        "logical qubits": logical,
        **footprint(logical, toffolis, p=p),
    }


def show(est: dict) -> None:
    """Print an estimate as a two-column table."""
    for k, v in est.items():
        if isinstance(v, float) and (abs(v) >= 1e5 or 0 < abs(v) < 1e-2):
            s = f"{v:.3g}"
        elif isinstance(v, float):
            s = f"{v:,.2f}"
        else:
            s = f"{v:,}"
        print(f"{k:<44} {s:>16}")
