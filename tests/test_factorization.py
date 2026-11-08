import pytest

from qcost.factorization import (
    df_lambda,
    df_logical_qubits,
    df_walk_toffolis,
    df_walk_toffolis_c39,
    double_factorize,
    reiher_femoco,
)
from qcost.hubbard import TABLE_IV, hubbard_estimate, hubbard_lambda
from qcost.walk import walk_calls

of_df = pytest.importorskip("openfermion.resource_estimates.df")


def test_reiher_femoco_matches_lee_et_al():
    h1, eri, _ = reiher_femoco()
    factors, L, LXi = double_factorize(eri, 0.00125)
    lam = df_lambda(h1, eri, factors)["total"]
    assert (L, LXi) == (360, 13031)
    assert round(lam, 1) == 294.8
    step = df_walk_toffolis(108, L, LXi, beth=16)["total"]
    calls = walk_calls(294.8, 0.001)
    assert (step, step * calls, df_logical_qubits(108, L, LXi, calls)["total"]) == (21753, 10073183463, 3725)


def test_step_list_matches_openfermion_and_c39_is_aleph_minus_1_above():
    for N, lam, L, LXi, beth, br1 in [(108, 294.8, 360, 13031, 16, 7), (152, 1171.2, 394, 20115, 20, 6)]:
        ours = df_walk_toffolis(N, L, LXi, beth=beth, b_r_first=br1)["total"]
        of = of_df.compute_cost(N, lam, 0.001, L, LXi, 10, beth, 20000)[0]
        assert of_df.compute_cost(N, lam, 0.001, L, LXi, 10, beth, of)[0] == ours
        # the closed form counts aleph - 1 = 9 more than the step list (at equal rotation bits)
        assert df_walk_toffolis_c39(N, L, LXi, beth=beth) - df_walk_toffolis(N, L, LXi, beth=beth)["total"] == 10 - 1


def test_hubbard_table_iv():
    assert hubbard_lambda(18, 1, 4) == 72  # 3 x 3 periodic, checked against OpenFermion's JW form
    for row in TABLE_IV.values():
        e = hubbard_estimate(row["N"])
        assert e["ancilla"] == row["ancilla"] and e["logical"] == row["logical"]
        assert abs(e["T"] / row["T"] - 1) < 0.1
