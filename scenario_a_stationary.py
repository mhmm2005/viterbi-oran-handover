"""
REAL-WORLD SCENARIO A - Stationary UE at the cell edge
==========================================================

Every scenario so far involved a UE that was ACTUALLY moving between
cells - there was always a real crossing happening. This scenario is
different and arguably more important for testing ping-pong
specifically: a UE that is essentially STATIONARY, parked right at (or
very near) the boundary between two cells - e.g., someone sitting in a
cafe that happens to be near the edge of coverage.

In this case the TRUE serving cell never changes (or changes at most
once, right at the start, then stays fixed) - any handover that
happens after that is, by definition, spurious. This is the purest
possible test of ping-pong specifically, stripped of any "did we
correctly detect a real transition" complexity.

Real-world relevance: this happens constantly - people sit at desks,
in cafes, on park benches, near cell edges for hours. If an algorithm
ping-pongs here, it's pure wasted signaling with zero benefit.
"""

import numpy as np

CELL_A_POS = 0.0
CELL_B_POS = 1000.0
TX_POWER_DBM = -60.0
PATH_LOSS_EXPONENT = 3.5
MIN_DISTANCE_M = 1.0

# UE parked slightly off-center so there IS a well-defined true answer,
# but close enough to the boundary that it's a genuinely hard case
UE_POSITION_X = 510.0   # 10m past the exact midpoint - Cell B is *barely* the true best cell
DURATION_S = 300         # 5 minutes stationary
NOISE_STD_DB = 10.0


def rsrp_from_distance(distance_m):
    d = np.maximum(distance_m, MIN_DISTANCE_M)
    return TX_POWER_DBM - PATH_LOSS_EXPONENT * 10 * np.log10(d)


def generate_stationary_data(seed=0):
    t = np.arange(0, DURATION_S, 1.0)
    n = len(t)

    dist_a = abs(UE_POSITION_X - CELL_A_POS)
    dist_b = abs(UE_POSITION_X - CELL_B_POS)
    true_rsrp_a = np.full(n, rsrp_from_distance(dist_a))
    true_rsrp_b = np.full(n, rsrp_from_distance(dist_b))

    rng = np.random.default_rng(seed)
    noisy_rsrp_a = true_rsrp_a + rng.normal(0, NOISE_STD_DB, size=n)
    noisy_rsrp_b = true_rsrp_b + rng.normal(0, NOISE_STD_DB, size=n)

    true_serving = 1 if true_rsrp_b[0] > true_rsrp_a[0] else 0  # constant - UE never truly moves

    return {
        "t": t, "noisy_rsrp_a": noisy_rsrp_a, "noisy_rsrp_b": noisy_rsrp_b,
        "true_rsrp_a": true_rsrp_a, "true_rsrp_b": true_rsrp_b,
        "true_serving_cell": np.full(n, true_serving),
    }


if __name__ == "__main__":
    from step2_threshold_handover import naive_threshold_handover, hysteresis_handover, analyze_handovers
    from step3_viterbi import viterbi_decode

    data = generate_stationary_data()
    t = data["t"]

    print(f"True RSRP gap (B - A): {data['true_rsrp_b'][0] - data['true_rsrp_a'][0]:.2f} dB "
          f"(small = genuinely hard case)")
    print(f"UE is stationary, true serving cell never changes: Cell {data['true_serving_cell'][0]}\n")

    sv_naive = naive_threshold_handover(data["noisy_rsrp_a"], data["noisy_rsrp_b"])
    r_naive = analyze_handovers(sv_naive, t)

    sv_hyst = hysteresis_handover(data["noisy_rsrp_a"], data["noisy_rsrp_b"])
    r_hyst = analyze_handovers(sv_hyst, t)

    diff = data["noisy_rsrp_b"] - data["noisy_rsrp_a"]
    sv_vit = viterbi_decode(diff)
    r_vit = analyze_handovers(sv_vit, t)

    print(f"{'Algorithm':<15}{'Total HO':>10}{'Ping-pongs':>12}")
    for name, r in [("Naive", r_naive), ("Hysteresis", r_hyst), ("Viterbi", r_vit)]:
        print(f"{name:<15}{r['total_handovers']:>10}{r['ping_pongs']:>12}")
