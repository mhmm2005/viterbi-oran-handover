"""
REAL-WORLD SCENARIO D - Heterogeneous network (macro + small cell)
=======================================================================

Every previous scenario assumed all cells have IDENTICAL transmit
power. Real networks are heterogeneous ("HetNet"): a high-power macro
cell providing wide-area coverage, with low-power small cells dropped
in at hotspots (shopping centers, stadiums, dense urban blocks) to add
capacity. A small cell might have 15-20dB less transmit power than the
macro cell, but because it's so much closer to nearby users, it can
still win the RSRP comparison in a small coverage bubble around itself.

This is a harder case because the SIZE of the "small cell wins" zone
is tiny (proportional to its lower power), so a UE walking near - but
not necessarily through - the small cell's bubble experiences a sharp,
narrow spike of ambiguity, very different from the smooth, symmetric
crossings in Scenario 1.
"""

import numpy as np

MACRO_POS = 0.0
MACRO_TX_DBM = -60.0        # macro cell reference power

SMALL_CELL_POS = 300.0      # small cell dropped in along the UE's path
SMALL_CELL_TX_DBM = -78.0   # 18dB weaker transmitter (typical macro-vs-small-cell gap)

PATH_LOSS_EXPONENT = 3.5
MIN_DISTANCE_M = 1.0
NOISE_STD_DB = 10.0

UE_START_X = -100.0
UE_END_X = 700.0
UE_SPEED_MPS = 1.4     # walking pace - small cells are usually a pedestrian/hotspot scenario
TIME_STEP_S = 1.0


def rsrp_from_distance(distance_m, tx_power_dbm):
    d = np.maximum(distance_m, MIN_DISTANCE_M)
    return tx_power_dbm - PATH_LOSS_EXPONENT * 10 * np.log10(d)


def generate_hetnet_data(seed=0):
    total_distance = UE_END_X - UE_START_X
    n = int((total_distance / UE_SPEED_MPS) / TIME_STEP_S)
    t = np.arange(n) * TIME_STEP_S
    x = UE_START_X + UE_SPEED_MPS * t

    dist_macro = np.abs(x - MACRO_POS)
    dist_small = np.abs(x - SMALL_CELL_POS)

    true_rsrp_macro = rsrp_from_distance(dist_macro, MACRO_TX_DBM)
    true_rsrp_small = rsrp_from_distance(dist_small, SMALL_CELL_TX_DBM)

    rng = np.random.default_rng(seed)
    noisy_rsrp_macro = true_rsrp_macro + rng.normal(0, NOISE_STD_DB, size=n)
    noisy_rsrp_small = true_rsrp_small + rng.normal(0, NOISE_STD_DB, size=n)

    true_serving = np.where(true_rsrp_small > true_rsrp_macro, 1, 0)  # 0=macro, 1=small

    return {
        "t": t, "x": x,
        "true_rsrp_a": true_rsrp_macro, "true_rsrp_b": true_rsrp_small,
        "noisy_rsrp_a": noisy_rsrp_macro, "noisy_rsrp_b": noisy_rsrp_small,
        "true_serving_cell": true_serving,
    }


if __name__ == "__main__":
    from step2_threshold_handover import naive_threshold_handover, hysteresis_handover, analyze_handovers
    from step3_viterbi import viterbi_decode

    data = generate_hetnet_data()
    t = data["t"]

    n_true = int(np.sum(np.diff(data["true_serving_cell"]) != 0))
    small_cell_zone_width = np.sum(data["true_serving_cell"] == 1) * UE_SPEED_MPS * TIME_STEP_S
    print(f"Small cell true coverage bubble width (along this path): ~{small_cell_zone_width:.0f}m")
    print(f"Ground-truth true handovers: {n_true}\n")

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
