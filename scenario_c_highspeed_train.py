"""
REAL-WORLD SCENARIO C - High-speed train (frequent real handovers)
========================================================================

A single fast crossing between two cells isn't actually the hard part
of high-speed mobility - the hard part is that real high-speed rail
deployments use many CLOSELY-SPACED small cells along the track (often
every 300-500m) specifically because coverage needs to be dense at
that speed. This means TRUE handovers happen frequently and
legitimately - roughly every few seconds at 300 km/h - which stresses
a very different assumption than our other scenarios: our P_SWITCH
prior assumed "switching cells is rare." Here it's not rare at all.

This tests whether the same P_SWITCH=0.02 (tuned for "rare" switching
in scenarios 1/2) still works when real handovers are legitimately
frequent, or whether it needs to be re-tuned for this regime.
"""

import numpy as np

TX_POWER_DBM = -60.0
PATH_LOSS_EXPONENT = 3.5
MIN_DISTANCE_M = 1.0
NOISE_STD_DB = 10.0

TRAIN_SPEED_MPS = 83.3    # 300 km/h
CELL_SPACING_M = 400.0    # small cells every 400m along the track (realistic for HSR)
N_CELLS = 15               # track covered by 15 consecutive cells
TRACK_LENGTH_M = CELL_SPACING_M * (N_CELLS - 1)
TIME_STEP_S = 1.0


def rsrp_from_distance(distance_m):
    d = np.maximum(distance_m, MIN_DISTANCE_M)
    return TX_POWER_DBM - PATH_LOSS_EXPONENT * 10 * np.log10(d)


def generate_train_data(seed=0):
    total_time_s = TRACK_LENGTH_M / TRAIN_SPEED_MPS
    n = int(total_time_s / TIME_STEP_S)
    t = np.arange(n) * TIME_STEP_S
    x = TRAIN_SPEED_MPS * t  # train position along the track

    cell_positions = np.arange(N_CELLS) * CELL_SPACING_M

    true_rsrp = np.zeros((n, N_CELLS))
    for c in range(N_CELLS):
        dist = np.abs(x - cell_positions[c])
        true_rsrp[:, c] = rsrp_from_distance(dist)

    rng = np.random.default_rng(seed)
    noisy_rsrp = true_rsrp + rng.normal(0, NOISE_STD_DB, size=true_rsrp.shape)

    true_serving_cell = np.argmax(true_rsrp, axis=1)

    return {"t": t, "x": x, "true_rsrp": true_rsrp, "noisy_rsrp": noisy_rsrp,
            "true_serving_cell": true_serving_cell, "cell_positions": cell_positions}


if __name__ == "__main__":
    from step5_multicell_baseline import (
        naive_multicell_handover, hysteresis_multicell_handover, analyze_handovers_multicell
    )
    from step6_multicell_viterbi import viterbi_decode_multicell

    data = generate_train_data()
    t = data["t"]
    n_true = int(np.sum(np.diff(data["true_serving_cell"]) != 0))

    print(f"Track: {N_CELLS} cells spaced {CELL_SPACING_M}m apart, {TRACK_LENGTH_M}m total")
    print(f"Train speed: {TRAIN_SPEED_MPS} m/s ({TRAIN_SPEED_MPS*3.6:.0f} km/h)")
    print(f"Total trip time: {t[-1]:.0f}s, samples: {len(t)}")
    print(f"Ground-truth true handovers: {n_true} "
          f"(~1 every {t[-1]/max(n_true,1):.1f}s - these are legitimately frequent)\n")

    sv_naive = naive_multicell_handover(data["noisy_rsrp"])
    r_naive = analyze_handovers_multicell(sv_naive, t)

    sv_hyst = hysteresis_multicell_handover(data["noisy_rsrp"])
    r_hyst = analyze_handovers_multicell(sv_hyst, t)

    # try our existing P_SWITCH=0.02 (tuned for "rare" switching) as-is first
    sv_vit_default = viterbi_decode_multicell(data["noisy_rsrp"], NOISE_STD_DB, p_switch=0.02)
    r_vit_default = analyze_handovers_multicell(sv_vit_default, t)

    print(f"{'Algorithm':<28}{'Total HO':>10}{'Ping-pongs':>12}")
    for name, r in [("Naive", r_naive), ("Hysteresis", r_hyst),
                     ("Viterbi (p_switch=0.02)", r_vit_default)]:
        print(f"{name:<28}{r['total_handovers']:>10}{r['ping_pongs']:>12}")
