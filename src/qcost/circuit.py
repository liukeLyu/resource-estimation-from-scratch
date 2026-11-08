"""A reversible circuit you can count and run.

Every circuit in this package is built from five gates on qubits labelled by integers:

* ``x(t)``, ``cx(c, t)``: Clifford, free in a surface code.
* ``ccx(a, b, t)``: the Toffoli, one non-Clifford gate.
* ``and_(a, b, t)``: a Toffoli onto a qubit known to be ``|0>`` (Gidney's temporary logical-AND).
  It costs one Toffoli (4 T) [Gidney, arXiv:1709.06648, Fig. 3].
* ``unand(a, b, t)``: undo an ``and_`` by measuring ``t`` in the X basis and fixing the sign with
  a CZ when the outcome is 1. Clifford plus measurement, so zero Toffolis [same figure].
* ``cswap(c, s, t)``: controlled swap, one Toffoli and two CNOTs.

The cost of a circuit is its Toffoli count: ``and_``, ``ccx`` and ``cswap`` each count one.

All of these gates map computational basis states to basis states (the sign fix of ``unand`` aside),
so a circuit can be checked on every input at once by pushing bit strings through it: ``run`` takes
a batch of inputs as a boolean array, one row per input, one column per qubit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TOFFOLI_GATES = ("ccx", "and", "cswap")


@dataclass
class Circuit:
    """A list of gates on ``n`` qubits; ``new_qubits`` grows ``n`` as ancillas are needed."""

    n: int = 0
    gates: list = field(default_factory=list)

    def new_qubits(self, k: int) -> list[int]:
        q = list(range(self.n, self.n + k))
        self.n += k
        return q

    def x(self, t):
        self.gates.append(("x", t))

    def cx(self, c, t):
        self.gates.append(("cx", c, t))

    def ccx(self, a, b, t):
        self.gates.append(("ccx", a, b, t))

    def and_(self, a, b, t):
        self.gates.append(("and", a, b, t))

    def unand(self, a, b, t):
        self.gates.append(("unand", a, b, t))

    def cswap(self, c, s, t):
        self.gates.append(("cswap", c, s, t))

    def extend(self, other: Circuit):
        self.gates.extend(other.gates)
        self.n = max(self.n, other.n)

    def toffolis(self) -> int:
        return sum(g[0] in TOFFOLI_GATES for g in self.gates)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for g in self.gates:
            out[g[0]] = out.get(g[0], 0) + 1
        return out

    def run(self, bits: np.ndarray) -> np.ndarray:
        """Apply the circuit to a batch of basis states, ``bits[i, q]`` = qubit ``q`` of input ``i``.

        Checks the promises the cheap gates rely on: ``and_`` needs its target at 0, and ``unand``
        needs its target equal to the AND it undoes.
        """
        s = np.array(bits, dtype=bool, copy=True)
        if s.shape[1] < self.n:
            s = np.hstack([s, np.zeros((s.shape[0], self.n - s.shape[1]), dtype=bool)])
        for g in self.gates:
            op = g[0]
            if op == "x":
                s[:, g[1]] ^= True
            elif op == "cx":
                s[:, g[2]] ^= s[:, g[1]]
            elif op == "ccx":
                s[:, g[3]] ^= s[:, g[1]] & s[:, g[2]]
            elif op == "and":
                assert not s[:, g[3]].any(), "and_ needs a target in |0>"
                s[:, g[3]] = s[:, g[1]] & s[:, g[2]]
            elif op == "unand":
                assert (s[:, g[3]] == (s[:, g[1]] & s[:, g[2]])).all(), "unand must undo an and_"
                s[:, g[3]] = False
            elif op == "cswap":
                c, a, b = g[1], g[2], g[3]
                swap = s[:, c] & (s[:, a] ^ s[:, b])
                s[:, a] ^= swap
                s[:, b] ^= swap
        return s


def write_int(bits: np.ndarray, qubits: list[int], values) -> None:
    """Store integers ``values`` (one per row) in ``qubits``, least significant bit first."""
    values = np.asarray(values)
    for j, q in enumerate(qubits):
        bits[:, q] = (values >> j) & 1


def read_int(bits: np.ndarray, qubits: list[int]) -> np.ndarray:
    """The integer held in ``qubits`` (least significant bit first), one per row."""
    return sum(bits[:, q].astype(np.int64) << j for j, q in enumerate(qubits))


def all_inputs(registers: list[list[int]], n: int) -> np.ndarray:
    """Every assignment of the given registers (others at 0): ``prod 2^len`` rows of ``n`` bits."""
    sizes = [len(r) for r in registers]
    grids = np.meshgrid(*[np.arange(2**k) for k in sizes], indexing="ij")
    bits = np.zeros((grids[0].size, n), dtype=bool)
    for r, g in zip(registers, grids, strict=True):
        write_int(bits, r, g.ravel())
    return bits
