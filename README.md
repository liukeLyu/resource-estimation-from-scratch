# resource-estimation-from-scratch

Fault-tolerant resource estimation for ground-state energies by qubitized phase estimation, built from scratch in NumPy: every Toffoli of the walk operator is a circuit that is run on every input and counted, first on the transverse-field Ising chain $H = -\sum_i X_iX_{i+1} - h\sum_i Z_i$, then on hydrogen chains, the Hubbard model and FeMoco. Checked against exact simulation, PennyLane's resource estimator, OpenFermion's `resource_estimates`, and the published tables of Babbush et al. (2018) and Lee et al. (2021), which it reproduces.

```python
import qcost

est = qcost.estimate(qcost.tfim(100), eps=0.01)    # 100 spins, ground energy to 0.01
qcost.show(est)
# lambda 199, 31 259 walk calls x 724 Toffolis = 2.3e7 Toffolis, 172 logical qubits,
# code distance 25, 440 000 physical qubits, 0.9 hours, each factor itemized
```

Bring your own Hamiltonian as a list of Pauli words, `qcost.PauliSum([(1.0, "XXI"), (0.5, "IZZ"), ...])`, or from OpenFermion with `qcost.exact.from_openfermion`. The one call is the wrong tool for a molecule: as a Pauli sum, H$_6$ in a minimal basis costs four times more Toffolis than in double-factorized form (`qcost.estimate_df`, notebook 05).

New to fault-tolerant costs? Start with [`00_start_here`](examples/00_start_here.ipynb): the whole estimate for a model you know, then for a molecule, with every factor named. Phone companion: the [pocket course](https://liukelyu.github.io/resource-estimation-from-scratch/docs/pocket-course.html).

| Step | What you learn | Notebook |
|---|---|---|
| 1 | Amplitude amplification: $1/\sqrt a$ rounds instead of $1/a$ tries, and the oblivious version that block encodings need; where it fails ($H/\lambda$ is not unitary) | [`01_amplitude_amplification`](examples/01_amplitude_amplification.ipynb) |
| 2 | Toffoli arithmetic: why only Toffolis count, the free uncomputation of a temporary AND, an adder ($n-1$), a comparator ($n$), rotations by a phase gradient | [`02_toffoli_arithmetic`](examples/02_toffoli_arithmetic.ipynb) |
| 3 | Loading data: unary iteration, QROM ($L-1$ Toffolis, any word length), select/swap and its best block size, undoing a lookup by measurement | [`03_qrom_select_swap`](examples/03_qrom_select_swap.ipynb) |
| 4 | PREPARE by coherent alias sampling, the qubitized walk and its spectrum $\pm\arccos(E/\lambda)$, the cost of one step | [`04_prepare_and_the_walk`](examples/04_prepare_and_the_walk.ipynb) |
| 5 | Double factorization by hand on hydrogen chains: two eigendecompositions, the one-norm, what truncation does to the energy | [`05_double_factorization`](examples/05_double_factorization.ipynb) |
| 6 | The surface code: code distance from the error budget, physical qubits, hours; every number of one estimate traced to its source | [`06_end_to_end_estimate`](examples/06_end_to_end_estimate.ipynb) |
| 7 | Reproduce the Hubbard estimate of Babbush et al.; what the model's structure saves | [`07_reproduce_hubbard`](examples/07_reproduce_hubbard.ipynb) |
| 8 | Reproduce the double-factorized FeMoco estimate of Lee et al., line by line | [`08_reproduce_femoco`](examples/08_reproduce_femoco.ipynb) |

## What you will see

* The adder ($n - 1$ Toffolis), comparator ($n$), QROM ($L - 1$) and select/swap QROM ($\lceil L/k\rceil + b(k-1)$, two fewer without a control) run correctly on every input; PennyLane's resource estimator gives the same counts for the adder and the select/swap lookup.
* The qubitized walk of a 4-spin Ising chain has every phase $\pm\arccos(E_k/\lambda)$ to $4\times10^{-16}$.
* One round of oblivious amplitude amplification turns a block encoding of $e^{i\varphi X}/2$ into $e^{i\varphi X}$ exactly, for every input state; on $H/\lambda$ the success amplitude depends on the state, so no fixed round count works.
* Double factorization of hydrogen chains: $\lambda$ is 2.9 times smaller than the Pauli sum's at 20 atoms, and the threshold $10^{-4}$ keeps the H$_6$ energy within $10^{-6}$ hartree of exact.
* Hubbard model, Babbush et al., Table IV: every qubit count exact, T counts within 1% ($6\times6$ to $10\times10$) and 5% ($20\times20$).
* FeMoco (Reiher et al. active space): $L = 360$, $L\Xi = 13\,031$, $\lambda = 294.8$ Ha, 21 753 Toffolis per step, $1.007\times10^{10}$ Toffolis on 3725 logical qubits, all exactly as in Lee et al., Table III; the angle lookups are 56% of each step.

## Install

```bash
pip install -e ".[dev,examples]"
pytest
```

Each notebook cell runs in seconds. `python scripts/make_results.py` remakes the small summaries in `results/` (about ten seconds on 4 cores). Conventions: qubit 0 is the leftmost letter of a Pauli word and the least significant bit of a register; two-electron integrals are in chemists' order, $(pq|rs)$, over spatial orbitals; energies are in the units of the Hamiltonian (hartree for molecules).

## References

* R. Babbush, C. Gidney, D. W. Berry, N. Wiebe, J. McClean, A. Paler, A. Fowler, H. Neven, *Encoding electronic spectra in quantum circuits with linear T complexity*, [arXiv:1805.03662](https://arxiv.org/abs/1805.03662); [Phys. Rev. X 8, 041015 (2018)](https://doi.org/10.1103/PhysRevX.8.041015)
* J. Lee, D. W. Berry, C. Gidney, W. J. Huggins, J. R. McClean, N. Wiebe, R. Babbush, *Even more efficient quantum computations of chemistry through tensor hypercontraction*, [arXiv:2011.03494](https://arxiv.org/abs/2011.03494); [PRX Quantum 2, 030305 (2021)](https://doi.org/10.1103/PRXQuantum.2.030305)
* V. von Burg, G. H. Low, T. Häner, D. S. Steiger, M. Reiher, M. Roetteler, M. Troyer, *Quantum computing enhanced computational catalysis*, [arXiv:2007.14460](https://arxiv.org/abs/2007.14460); [Phys. Rev. Research 3, 033055 (2021)](https://doi.org/10.1103/PhysRevResearch.3.033055)
* D. W. Berry, C. Gidney, M. Motta, J. R. McClean, R. Babbush, *Qubitization of arbitrary basis quantum chemistry leveraging sparsity and low rank factorization*, [arXiv:1902.02134](https://arxiv.org/abs/1902.02134); [Quantum 3, 208 (2019)](https://doi.org/10.22331/q-2019-12-02-208)
* G. H. Low, V. Kliuchnikov, L. Schaeffer, *Trading T gates for dirty qubits in state preparation and unitary synthesis*, [arXiv:1812.00954](https://arxiv.org/abs/1812.00954); [Quantum 8, 1375 (2024)](https://doi.org/10.22331/q-2024-06-17-1375)
* G. H. Low, I. L. Chuang, *Hamiltonian simulation by qubitization*, [arXiv:1610.06546](https://arxiv.org/abs/1610.06546); [Quantum 3, 163 (2019)](https://doi.org/10.22331/q-2019-07-12-163)
* C. Gidney, *Halving the cost of quantum addition*, [arXiv:1709.06648](https://arxiv.org/abs/1709.06648); [Quantum 2, 74 (2018)](https://doi.org/10.22331/q-2018-06-18-74)
* Y. R. Sanders, D. W. Berry, P. C. S. Costa, L. W. Tessler, N. Wiebe, C. Gidney, H. Neven, R. Babbush, *Compilation of fault-tolerant quantum heuristics for combinatorial optimization*, [arXiv:2007.07391](https://arxiv.org/abs/2007.07391); [PRX Quantum 1, 020312 (2020)](https://doi.org/10.1103/PRXQuantum.1.020312)
* N. J. Ross, P. Selinger, *Optimal ancilla-free Clifford+T approximation of z-rotations*, [arXiv:1403.2975](https://arxiv.org/abs/1403.2975); Quantum Inf. Comput. 16, 901 (2016)
* V. Kliuchnikov, K. Lauter, R. Minko, A. Paetznick, C. Petit, *Shorter quantum circuits via single-qubit gate approximation*, [arXiv:2203.10064](https://arxiv.org/abs/2203.10064); [Quantum 7, 1208 (2023)](https://doi.org/10.22331/q-2023-12-18-1208)
* G. Brassard, P. Høyer, M. Mosca, A. Tapp, *Quantum amplitude amplification and estimation*, [arXiv:quant-ph/0005055](https://arxiv.org/abs/quant-ph/0005055); [Contemp. Math. 305, 53 (2002)](https://doi.org/10.1090/conm/305/05215)
* D. W. Berry, A. M. Childs, R. Cleve, R. Kothari, R. D. Somma, *Exponential improvement in precision for simulating sparse Hamiltonians*, [arXiv:1312.1414](https://arxiv.org/abs/1312.1414); [STOC 2014, 283](https://doi.org/10.1145/2591796.2591854)
* D. W. Berry, A. M. Childs, R. Cleve, R. Kothari, R. D. Somma, *Simulating Hamiltonian dynamics with a truncated Taylor series*, [arXiv:1412.4687](https://arxiv.org/abs/1412.4687); [Phys. Rev. Lett. 114, 090502 (2015)](https://doi.org/10.1103/PhysRevLett.114.090502)
* C. Gidney, A. G. Fowler, *Efficient magic state factories with a catalyzed |CCZ⟩ to 2|T⟩ transformation*, [arXiv:1812.01238](https://arxiv.org/abs/1812.01238); [Quantum 3, 135 (2019)](https://doi.org/10.22331/q-2019-04-30-135)
* A. G. Fowler, C. Gidney, *Low overhead quantum computation using lattice surgery*, [arXiv:1808.06709](https://arxiv.org/abs/1808.06709)
* C. Gidney, M. Ekerå, *How to factor 2048 bit RSA integers in 8 hours using 20 million noisy qubits*, [arXiv:1905.09749](https://arxiv.org/abs/1905.09749); [Quantum 5, 433 (2021)](https://doi.org/10.22331/q-2021-04-15-433)
* D. Litinski, *A Game of Surface Codes: Large-scale quantum computing with lattice surgery*, [arXiv:1808.02892](https://arxiv.org/abs/1808.02892); [Quantum 3, 128 (2019)](https://doi.org/10.22331/q-2019-03-05-128)
* R. Babbush, J. McClean, M. Newman, C. Gidney, S. Boixo, H. Neven, *Focus beyond quadratic speedups for error-corrected quantum advantage*, [arXiv:2011.04149](https://arxiv.org/abs/2011.04149); [PRX Quantum 2, 010103 (2021)](https://doi.org/10.1103/PRXQuantum.2.010103)
* M. Reiher, N. Wiebe, K. M. Svore, D. Wecker, M. Troyer, *Elucidating reaction mechanisms on quantum computers*, [arXiv:1605.03590](https://arxiv.org/abs/1605.03590); [PNAS 114, 7555 (2017)](https://doi.org/10.1073/pnas.1619152114)
* Z. Li, J. Li, N. S. Dattani, C. J. Umrigar, G. K.-L. Chan, *The electronic complexity of the ground-state of the FeMo cofactor of nitrogenase as relevant to quantum simulations*, [arXiv:1809.10307](https://arxiv.org/abs/1809.10307); [J. Chem. Phys. 150, 024302 (2019)](https://doi.org/10.1063/1.5063376)
* I. D. Kivlichan, C. Gidney, D. W. Berry, N. Wiebe, J. McClean, W. Sun, Z. Jiang, N. Rubin, A. Fowler, et al., *Improved fault-tolerant quantum simulation of condensed-phase correlated electrons via Trotterization*, [arXiv:1902.10673](https://arxiv.org/abs/1902.10673); [Quantum 4, 296 (2020)](https://doi.org/10.22331/q-2020-07-16-296)
* E. Lieb, T. Schultz, D. Mattis, *Two soluble models of an antiferromagnetic chain*, [Ann. Phys. 16, 407 (1961)](https://doi.org/10.1016/0003-4916(61)90115-4)
