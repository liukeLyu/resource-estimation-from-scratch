"""Fault-tolerant resource estimation from scratch: build each piece of qubitized phase estimation, count it, check it.

``estimate`` is the one call: a Hamiltonian in, Toffolis, qubits and hours out. The modules
underneath, in the order the notebooks build them: ``circuit`` (a reversible circuit you can
count and run), ``amplify``, ``arithmetic``, ``qrom``, ``alias``, ``walk``, ``factorization``,
``surface_code``, ``hubbard``.
"""

from .estimate import estimate, estimate_df, show
from .exact import PauliSum, spectrum, tfim, tfim_open_energy

__all__ = ["PauliSum", "estimate", "estimate_df", "show", "spectrum", "tfim", "tfim_open_energy"]
