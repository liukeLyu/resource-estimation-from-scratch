"""From logical qubits and Toffolis to physical qubits and hours, in a surface code.

A simple model with the standard rules of thumb:

* A logical qubit is a ``d x d`` patch of about ``2 (d + 1)^2`` physical qubits (data, measurement
  and spacing) [Gidney and Ekera, arXiv:1905.09749, Sec. 2.14].
* Each logical qubit fails per round of error correction with probability
  ``p_L = 0.1 (100 p)^((d + 1) / 2)`` at physical error rate ``p`` [Fowler and Gidney,
  arXiv:1808.06709, Sec. XV; Litinski, arXiv:1808.02892, Eq. 10]. Below ``p = 1%`` each step of 2
  in ``d`` divides it by ``1 / (100 p)``: ten at ``p = 10^-3``.
* Toffolis come from a CCZ factory of ``12 d x 6 d`` that makes one CCZ state every ``5.5 d``
  rounds [Gidney and Fowler, arXiv:1812.01238]. Clifford gates are cheap by comparison, so the run
  time is the Toffoli count times the time per Toffoli.
* One round takes 1 microsecond [same papers].

``footprint`` picks the smallest odd ``d`` that keeps the expected number of logical errors over
the whole run below ``budget``. Routing space is taken as half the data area again, as in
OpenFermion's ``cost_estimator`` (which also searches over factory designs and is a cross-check).
"""

from __future__ import annotations

import numpy as np


def logical_error_per_round(p: float, d: int) -> float:
    """``0.1 (100 p)^((d + 1) / 2)`` [arXiv:1808.06709, Sec. XV]."""
    return 0.1 * (100 * p) ** ((d + 1) / 2)


def footprint(
    logical_qubits: int,
    toffolis: float,
    p: float = 1e-3,
    cycle_seconds: float = 1e-6,
    budget: float = 0.01,
    factories: int = 1,
    routing: float = 0.5,
) -> dict:
    """Code distance, physical qubits and wall-clock time for a run of ``toffolis`` on ``logical_qubits``."""
    patches = int(np.ceil(logical_qubits * (1 + routing)))
    for d in range(3, 101, 2):
        rounds = toffolis * 5.5 * d / factories
        if patches * rounds * logical_error_per_round(p, d) < budget:
            break
    data = patches * 2 * (d + 1) ** 2
    factory = factories * 2 * (12 * d) * (6 * d)
    seconds = rounds * cycle_seconds
    return {
        "distance": d,
        "physical qubits": data + factory,
        "data qubits": data,
        "factory qubits": factory,
        "seconds per Toffoli": 5.5 * d * cycle_seconds / factories,
        "hours": seconds / 3600,
    }
