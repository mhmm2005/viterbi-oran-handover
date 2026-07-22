"""
REAL-WORLD SCENARIO B - Sudden blockage (tunnel / building shadowing)
=========================================================================

Every previous scenario modeled signal changes as GRADUAL (smooth
path-loss curves) plus Gaussian noise. But real blockage events are
different in character: a UE can be cruising along fine, then suddenly
enters a tunnel, drives behind a large building, or goes through a
dense urban canyon - and the serving cell's signal drops by 20-30+ dB
almost instantly, then recovers just as suddenly when the obstruction
clears. This is a STEP CHANGE, not a smooth curve.

This is an important stress test because it's the opposite of a real
handover: the true best cell hasn't actually changed (the network
topology didn't move), but the SERVING cell's signal has been
temporarily and severely degraded, sometimes below a further-away but
unobstructed neighbor. A good algorithm should ideally still avoid
handing over if the blockage is brief (because handing over into a
tunnel/blockage-adjacent cell during a transient dip, only to hand
back out a few seconds later, is exactly the kind of wasted signaling
we're trying to avoid) - but should NOT be so stubborn that it refuses
a handover during a long/permanent blockage.

We simulate: a UE driving in Cell A's territory the whole time
(never actually crosses into B's territory), but experiences a
short deep-fade blockage event partway through.
"""

import numpy as np

CELL_A_POS = 0.0
CELL_B_POS = 1000.0
TX_POWER_DBM = -60.0
PATH_LOSS_EXPONENT = 3.5
MIN_DISTANCE_M = 1.0
NOISE_STD_DB = 10.0

UE_POSITION_X = 200.0     # stays well within Cell A's territory the whole time
DURATION_S = 200

BLOCKAGE_START_S = 90
BLOCKAGE_DURATION_S = 8     # a short blockage (e.g. underpass) - should NOT trigger handover
BLOCKAGE_DEPTH_DB = 30      # severe attenuation while blocked


def rsrp_from_distance(distance_m):
    d = np.maximum(distance_m, MIN_DISTANCE_M)
    return TX_POWER_DBM - PATH_LOSS_EXPONENT * 10 * np.log10(d)


def generate_blockage_data(seed=0, blockage_duration_s=BLOCKAGE_DURATION_S):
    t = np.arange(0, DURATION_S, 1.0)
    n = len(t)

    dist_a = abs(UE_POSITION_X - CELL_A_POS)
    dist_b = abs(UE_POSITION_X - CELL_B_POS)
    true_rsrp_a = np.full(n, rsrp_from_distance(dist_a))
    true_rsrp_b = np.full(n, rsrp_from_distance(dist_b))

    # apply the blockage: a step-function dip in Cell A's signal only
    # (the blockage is specific to the line-of-sight path to Cell A;
    # Cell B, being far away in a different direction, is unaffected -
    # this is realistic, e.g. driving under a bridge that blocks one
    # direction but not another)
    blockage_mask = (t >= BLOCKAGE_START_S) & (t < BLOCKAGE_START_S + blockage_duration_s)
    true_rsrp_a_with_blockage = true_rsrp_a.copy()
    true_rsrp_a_with_blockage[blockage_mask] -= BLOCKAGE_DEPTH_DB

    rng = np.random.default_rng(seed)
    noisy_rsrp_a = true_rsrp_a_with_blockage + rng.normal(0, NOISE_STD_DB, size=n)
    noisy_rsrp_b = true_rsrp_b + rng.normal(0, NOISE_STD_DB, size=n)

    # ground truth: Cell A is ALWAYS the geometrically correct serving
    # cell (the blockage is a temporary link impairment, not a real
    # change in which cell is closest/best) - true_serving_cell is
    # constant regardless of the blockage
    true_serving_cell = np.zeros(n, dtype=int)  # always Cell A

    return {
        "t": t, "noisy_rsrp_a": noisy_rsrp_a, "noisy_rsrp_b": noisy_rsrp_b,
        "true_rsrp_a": true_rsrp_a_with_blockage, "true_rsrp_b": true_rsrp_b,
        "true_serving_cell": true_serving_cell, "blockage_mask": blockage_mask,
    }


if __name__ == "__main__":
    from step2_threshold_handover import naive_threshold_handover, hysteresis_handover, analyze_handovers
    from step3_viterbi import viterbi_decode

    data = generate_blockage_data()
    t = data["t"]

    print(f"UE stays in Cell A territory the whole time.")
    print(f"Blockage: {BLOCKAGE_DURATION_S}s dip of {BLOCKAGE_DEPTH_DB}dB starting at t={BLOCKAGE_START_S}s\n")

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

    print("\nViterbi handover events:")
    for time_val, (frm, to) in zip(r_vit["handover_times"], r_vit["handover_from_to"]):
        print(f"  t={time_val:6.1f}s   Cell {frm} -> Cell {to}")
