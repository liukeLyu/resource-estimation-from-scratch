"""Loading classical data into a quantum register: unary iteration, QROM and select/swap.

* ``unary_iteration``: visit every index ``l < L`` of an address register, raising a flag qubit
  that is 1 exactly when the address equals ``l``. The flags are the nodes of a binary tree: a
  left child is an AND of its parent and a negated address bit (one Toffoli), its right sibling
  is then one CNOT away (parent XOR left child), and the AND is undone by measurement for free.
  ``L - 1`` Toffolis when controlled, one fewer without a control [Babbush et al.,
  arXiv:1805.03662, Sec. III A, Fig. 7: ``4L - 4`` T gates].
* ``qrom``: unary iteration that writes ``data[l]`` with CNOTs from the flag. The word length is
  free: CNOTs are Clifford. Same ``L - 1`` Toffolis [same paper, Fig. 10].
* ``select_swap_qrom``: iterate over blocks of ``k`` words, write a whole block into ``k``
  registers, then move the wanted word to the front with a network of controlled swaps. Fewer
  Toffolis, ``ceil(L/k) + b (k - 1)``, for ``b (k - 1)`` more qubits; best near ``k = sqrt(L/b)``
  [Low, Kliuchnikov, Schaeffer, arXiv:1812.00954; Berry et al., arXiv:1902.02134, App. A].
* Uncomputing a QROM by measuring its output and fixing signs costs ``ceil(L/k) + k`` Toffolis,
  best near ``k = sqrt(L)`` [Berry et al., arXiv:1902.02134, App. C].
"""

from __future__ import annotations

import numpy as np

from .circuit import Circuit


def n_bits(L: int) -> int:
    """Address bits needed to label ``L`` items."""
    return max(1, int(np.ceil(np.log2(L))))


def unary_iteration(circ: Circuit, address: list[int], L: int, leaf, control: int | None = None) -> None:
    """Append unary iteration over ``l < L`` to ``circ``; ``leaf(l, flag)`` adds the gates for item ``l``.

    ``flag`` is a qubit that is 1 exactly when ``address == l`` (and ``control`` is 1), or ``None``
    when the item needs no condition (``L = 1`` without control). Addresses ``>= L`` are assumed
    never to occur, which lets a node with an empty right half pass its flag straight down.
    """
    m = len(address)
    anc = circ.new_qubits(m)  # one work qubit per tree level, reused across siblings

    def visit(flag, level, lo):
        if level == 0:
            leaf(lo, flag)
            return
        q, half = address[level - 1], 2 ** (level - 1)
        if lo + half >= L:  # only the left half holds real items
            visit(flag, level - 1, lo)
            return
        if flag is None:  # uncontrolled root: the address bit itself is the flag, no Toffoli
            circ.x(q)
            visit(q, level - 1, lo)
            circ.x(q)
            visit(q, level - 1, lo + half)
            return
        t = anc[level - 1]
        circ.x(q)
        circ.and_(flag, q, t)  # t = flag AND (bit is 0)
        circ.x(q)
        visit(t, level - 1, lo)
        circ.cx(flag, t)  # t = flag AND (bit is 1)
        visit(t, level - 1, lo + half)
        circ.unand(flag, q, t)

    visit(control, m, 0)


def qrom(data, b: int, controlled: bool = False) -> tuple[Circuit, list[int], list[int], int | None]:
    """Circuit writing ``data[address]`` (``b``-bit integers) into an output register.

    Returns ``(circuit, address, output, control)``; ``control`` is ``None`` when uncontrolled.
    """
    L = len(data)
    c = Circuit()
    address, out = c.new_qubits(n_bits(L)), c.new_qubits(b)
    ctrl = c.new_qubits(1)[0] if controlled else None

    def leaf(l, flag):
        for j in range(b):
            if (int(data[l]) >> j) & 1:
                c.x(out[j]) if flag is None else c.cx(flag, out[j])

    unary_iteration(c, address, L, leaf, ctrl)
    return c, address, out, ctrl


def select_swap_qrom(data, b: int, k: int) -> tuple[Circuit, list[int], list[int]]:
    """Select/swap QROM with ``k`` (a power of 2) words per block; the word lands in the first register.

    Returns ``(circuit, address, output)``. The other ``k - 1`` registers end holding the rest of
    the block (garbage, cleaned up later by running the circuit backwards or by measurement).
    """
    L = len(data)
    s = int(np.log2(k))
    assert 2**s == k, "k must be a power of 2"
    n_blocks = int(np.ceil(L / k))
    c = Circuit()
    low = c.new_qubits(s)
    high = c.new_qubits(n_bits(n_blocks)) if n_blocks > 1 else []
    regs = [c.new_qubits(b) for _ in range(k)]

    def leaf(B, flag):
        for j in range(k):
            l = B * k + j
            if l < L:
                for bit in range(b):
                    if (int(data[l]) >> bit) & 1:
                        c.x(regs[j][bit]) if flag is None else c.cx(flag, regs[j][bit])

    if n_blocks > 1:
        unary_iteration(c, high, n_blocks, leaf)
    else:
        leaf(0, None)
    # swap network: low bit 0 moves the odd words onto the even ones, bit 1 every 4th onto every 4th, ...
    for j in range(s):
        step = 2**j
        for i in range(0, k, 2 * step):
            for bit in range(b):
                c.cswap(low[j], regs[i][bit], regs[i + step][bit])
    return c, low + high, regs[0]


# Cost formulas


def qrom_toffolis(L: int, controlled: bool = True) -> int:
    """Toffolis of a QROM (or SELECT) by unary iteration over ``L`` items [arXiv:1805.03662]."""
    return max(0, L - 1 if controlled else L - 2)


def qroam_toffolis(L: int, b: int, k: int) -> int:
    """Select/swap QROM: ``ceil(L/k) + b (k - 1)`` Toffolis [Berry et al., arXiv:1902.02134, App. A]."""
    return int(np.ceil(L / k)) + b * (k - 1)


def qroam_uncompute_toffolis(L: int, k: int) -> int:
    """Measurement-based uncomputation of a QROM: ``ceil(L/k) + k`` Toffolis [same paper, App. C]."""
    return int(np.ceil(L / k)) + k


def best_k(cost, kmax: int = 2**16) -> tuple[int, int]:
    """The power of 2 ``k`` minimizing ``cost(k)``; returns ``(k, cost(k))``."""
    ks = [2**j for j in range(int(np.log2(kmax)) + 1)]
    vals = [cost(k) for k in ks]
    i = int(np.argmin(vals))
    return ks[i], vals[i]
