"""Full-size runs behind the notebook figures, saved as small JSON summaries in results/.

    python scripts/make_results.py              # every result, spread over all CPU cores
    python scripts/make_results.py hydrogen_chains

Each notebook runs a seconds-long live version of a result, then loads the full one from results/.
Everything is deterministic. About ten seconds on 4 cores, set by the longest hydrogen chains (the Pauli
one-norm needs the full Jordan-Wigner Hamiltonian). Every other result in the notebooks runs live.
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from qcost.factorization import df_lambda, double_factorize, h_chain, pauli_lambda

RESULTS = Path(__file__).resolve().parent.parent / "results"


def hydrogen_chain_run(n_atoms, thresh=1e-4):
    h1, eri = h_chain(n_atoms)
    factors, L, LXi = double_factorize(eri, thresh)
    return {
        "atoms": n_atoms,
        "lambda_pauli": pauli_lambda(h1, eri),
        "lambda_df": df_lambda(h1, eri, factors)["total"],
        "L": L,
        "LXi": LXi,
    }


RESULTS_SPEC = {
    "hydrogen_chains": ([(hydrogen_chain_run, (n,)) for n in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)], lambda rows: rows),
}


def _call(task):
    f, args = task
    return f(*args)


def make(name, pool):
    tasks, assemble = RESULTS_SPEC[name]
    t0 = time.time()
    rows = list(pool.map(_call, tasks))
    out = assemble(rows)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"{name}: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    names = sys.argv[1:] or list(RESULTS_SPEC)
    np.random.seed(0)
    with ProcessPoolExecutor() as pool:
        for name in names:
            make(name, pool)
