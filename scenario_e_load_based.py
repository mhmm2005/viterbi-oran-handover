"""
REAL-WORLD SCENARIO E - Load-based handover (a genuine limitation)
========================================================================

Every scenario so far assumed handover decisions are (and should be)
based purely on signal quality (RSRP). But real networks also hand
over UEs for LOAD BALANCING reasons: Cell A might have the strongest
signal for a given UE, but if Cell A is congested (too many users,
running low on resource blocks), the network may deliberately hand the
UE to a slightly weaker but LESS LOADED neighbor cell, to balance
traffic and maintain better overall quality of service.

This is fundamentally NOT a signal-noise problem - it's a scheduling/
resource-management decision layered on top of signal measurements.
Our Viterbi model only ever sees noisy RSRP; it has no visibility into
cell load. This scenario demonstrates, honestly, what happens when we
apply our RSRP-only Viterbi to a situation where the "correct" handover
doesn't actually follow the RSRP signal at all.

This is included specifically to show the boundary of what this
technique solves: it eliminates NOISE-DRIVEN ping-pong, but it doesn't
(and can't, on its own) replace network-directed, load-aware handover
decisions. A real deployment would need to combine this smoothed
RSRP-based state estimate with a SEPARATE load-balancing policy layer
- not replace it.
"""

import numpy as np

CELL_A_POS = 0.0
CELL_B_POS = 1000.0
TX_POWER_DBM = -60.0
PATH_LOSS_EXPONENT = 3.5
MIN_DISTANCE_M = 1.0
NOISE_STD_DB = 10.0

UE_POSITION_X = 200.0   # UE stays well within Cell A's natural RSRP-best territory
DURATION_S = 200

# The network decides, for LOAD reasons, to force a handover to Cell B
# for a period, even though Cell A remains the RSRP-best cell the whole time.
LOAD_FORCED_HANDOVER_START_S = 80
LOAD_FORCED_HANDOVER_END_S = 140


def rsrp_from_distance(distance_m):
    d = np.maximum(distance_m, MIN_DISTANCE_M)
    return TX_POWER_DBM - PATH_LOSS_EXPONENT * 10 * np.log10(d)


def generate_load_based_data(seed=0):
    t = np.arange(0, DURATION_S, 1.0)
    n = len(t)

    dist_a = abs(UE_POSITION_X - CELL_A_POS)
    dist_b = abs(UE_POSITION_X - CELL_B_POS)
    true_rsrp_a = np.full(n, rsrp_from_distance(dist_a))
    true_rsrp_b = np.full(n, rsrp_from_distance(dist_b))

    rng = np.random.default_rng(seed)
    noisy_rsrp_a = true_rsrp_a + rng.normal(0, NOISE_STD_DB, size=n)
    noisy_rsrp_b = true_rsrp_b + rng.normal(0, NOISE_STD_DB, size=n)

    # The NETWORK-DESIRED serving cell (what SHOULD happen operationally,
    # given load balancing) is different from the RSRP-best cell during
    # the forced window - this is the key mismatch this scenario tests.
    network_desired_serving = np.zeros(n, dtype=int)
    forced_mask = (t >= LOAD_FORCED_HANDOVER_START_S) & (t < LOAD_FORCED_HANDOVER_END_S)
    network_desired_serving[forced_mask] = 1  # Cell B, for load reasons only

    rsrp_best_cell = np.zeros(n, dtype=int)  # Cell A is RSRP-best the entire time in this scenario

    return {
        "t": t, "noisy_rsrp_a": noisy_rsrp_a, "noisy_rsrp_b": noisy_rsrp_b,
        "network_desired_serving": network_desired_serving,
        "rsrp_best_cell": rsrp_best_cell,
    }


if __name__ == "__main__":
    from step3_viterbi import viterbi_decode
    from step2_threshold_handover import analyze_handovers

    data = generate_load_based_data()
    t = data["t"]
    diff = data["noisy_rsrp_b"] - data["noisy_rsrp_a"]
    sv_vit = viterbi_decode(diff)

    print("RSRP-only Viterbi has NO visibility into network load.")
    print(f"Network wants UE on Cell B from t={LOAD_FORCED_HANDOVER_START_S}s to "
          f"t={LOAD_FORCED_HANDOVER_END_S}s (load balancing), but Cell A remains "
          f"RSRP-best the entire time.\n")

    print(f"{'t':>5} {'network_wants':>14} {'viterbi_decoded':>16} {'match?':>8}")
    mismatch_count = 0
    for i in range(0, len(t), 15):
        match = "YES" if sv_vit[i] == data["network_desired_serving"][i] else "no"
        if sv_vit[i] != data["network_desired_serving"][i]:
            mismatch_count += 1
        print(f"{t[i]:5.0f} {data['network_desired_serving'][i]:14d} {sv_vit[i]:16d} {match:>8}")

    total_mismatch = np.sum(sv_vit != data["network_desired_serving"])
    print(f"\nTotal timesteps where Viterbi's RSRP-only decision differs from "
          f"the network's load-driven intent: {total_mismatch}/{len(t)} "
          f"({100*total_mismatch/len(t):.0f}%)")
    print("\nThis is EXPECTED and illustrates the boundary of this technique: "
          "Viterbi correctly stays on the RSRP-best cell throughout, because "
          "that's all it was designed to optimize for. It has no mechanism to "
          "know it should temporarily move to a worse-signal cell for load "
          "balancing - that requires a separate signal (load/congestion info) "
          "layered on top.")
