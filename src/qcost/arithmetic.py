"""Toffoli arithmetic: an adder, a comparator, and rotations by adding into a phase gradient.

* ``adder``: ``b <- a + b mod 2^n`` with ``n - 1`` Toffolis. A ripple of carries, each carry
  computed with one temporary AND and uncomputed by measurement for free [Gidney,
  arXiv:1709.06648, Fig. 1: ``4n - 4`` T gates, i.e. ``n - 1`` Toffolis].
* ``comparator``: a flag set to ``a < b`` with ``n`` Toffolis: the carry out of
  ``(2^n - 1 - a) + b`` is 1 exactly when ``b > a``. Only the carries are computed (``n`` ANDs),
  the flag is copied out, and the carries are uncomputed by measurement. Derived here and checked
  on every input in ``tests/test_arithmetic.py``.
* ``phase_gradient_rotation``: a rotation by an angle held in a ``b``-bit register costs one
  ``b``-bit addition into the phase-gradient state, about ``b`` Toffolis instead of a synthesized
  rotation of ``3 log2(1/eps)`` T gates per angle [Gidney, arXiv:1709.06648; Sanders et al.,
  arXiv:2007.07391, App. A, give ``b - 2`` Toffolis].
"""

from __future__ import annotations

import numpy as np

from .circuit import Circuit


def adder(n: int) -> tuple[Circuit, list[int], list[int]]:
    """Circuit for ``b <- a + b mod 2^n``; returns ``(circuit, a, b)`` (registers least significant first)."""
    c = Circuit()
    a, b = c.new_qubits(n), c.new_qubits(n)
    carry = [None] + c.new_qubits(n - 1)  # carry[i] = carry into bit i, carry[0] = 0
    # forward: compute carries; a_i, b_i temporarily hold a_i ^ c_i, b_i ^ c_i
    for i in range(n - 1):
        if i == 0:
            c.and_(a[0], b[0], carry[1])
        else:
            c.cx(carry[i], a[i])
            c.cx(carry[i], b[i])
            c.and_(a[i], b[i], carry[i + 1])
            c.cx(carry[i], carry[i + 1])
    # top bit: no carry out needed
    if n > 1:
        c.cx(carry[n - 1], b[n - 1])
    c.cx(a[n - 1], b[n - 1])
    # backward: uncompute carries for free, leave the sum bits in b
    for i in range(n - 2, -1, -1):
        if i == 0:
            c.unand(a[0], b[0], carry[1])
        else:
            c.cx(carry[i], carry[i + 1])
            c.unand(a[i], b[i], carry[i + 1])
            c.cx(carry[i], a[i])
        c.cx(a[i], b[i])
    return c, a, b


def comparator(n: int) -> tuple[Circuit, list[int], list[int], int]:
    """Circuit setting ``flag ^= (a < b)`` for ``n``-bit registers; returns ``(circuit, a, b, flag)``."""
    c = Circuit()
    a, b = c.new_qubits(n), c.new_qubits(n)
    (flag,) = c.new_qubits(1)
    carry = [None] + c.new_qubits(n)  # carry[i+1] = carry out of bit i of (~a) + b
    for q in a:
        c.x(q)  # a -> ~a = 2^n - 1 - a
    undo = []
    for i in range(n):
        if i == 0:
            c.and_(a[0], b[0], carry[1])
            undo.append([("unand", a[0], b[0], carry[1])])
        else:
            c.cx(carry[i], a[i])
            c.cx(carry[i], b[i])
            c.and_(a[i], b[i], carry[i + 1])
            c.cx(carry[i], carry[i + 1])
            undo.append(
                [("cx", carry[i], carry[i + 1]), ("unand", a[i], b[i], carry[i + 1]),
                 ("cx", carry[i], b[i]), ("cx", carry[i], a[i])]
            )
    c.cx(carry[n], flag)
    for step in reversed(undo):
        c.gates.extend(step)
    for q in a:
        c.x(q)
    return c, a, b, flag


def adder_toffolis(n: int) -> int:
    """Toffolis of the ``n``-bit adder above (Gidney, arXiv:1709.06648)."""
    return n - 1


def comparator_toffolis(n: int) -> int:
    """Toffolis of the ``n``-bit comparator above (one AND per carry)."""
    return n


# Rotations from a phase gradient


def phase_gradient_state(b: int) -> np.ndarray:
    """``|phi> = 2^(-b/2) sum_k exp(-2 pi i k / 2^b) |k>``, made once and reused for every rotation."""
    k = np.arange(2**b)
    return np.exp(-2j * np.pi * k / 2**b) / np.sqrt(2**b)


def add_into(state: np.ndarray, theta: int, b: int) -> np.ndarray:
    """Basis action of ``|k> -> |k + theta mod 2^b>`` on a state of the gradient register."""
    return np.roll(state, theta % 2**b)


def phase_gradient_rotation(theta: int, b: int) -> complex:
    """Phase picked up when ``theta`` is added into the gradient state: ``exp(2 pi i theta / 2^b)``.

    Adding is a permutation, yet the gradient state is its eigenvector: shifting ``k`` by ``theta``
    multiplies every amplitude by the same phase. Controlled on a qubit, this is a Z rotation by
    ``2 pi theta / 2^b`` that costs one ``b``-bit addition.
    """
    phi = phase_gradient_state(b)
    out = add_into(phi, theta, b)
    return complex(np.vdot(phi, out))


def rotation_synthesis_t(eps: float) -> float:
    """T gates to synthesize one Z rotation to operator-norm error ``eps``, about ``3 log2(1/eps)``.

    Typical count of Ross and Selinger [arXiv:1403.2975]; their worst case is ``4 log2(1/eps) + K``.
    Repeat-until-success and mixed fallback circuits do better on average [Kliuchnikov et al.,
    arXiv:2203.10064: ``0.53 log2(1/eps) + 4.86`` mean, diamond norm].
    """
    return 3 * np.log2(1 / eps)
