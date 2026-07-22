"""
STEP 5 - Multi-cell baseline handover algorithms
====================================================

Same two baselines as Scenario 1, generalized from 2 cells to N cells:

  1. NAIVE: at every sample, serve whichever cell has the highest
     noisy RSRP right now.

  2. HYSTERESIS: only switch away from the current serving cell if
     some OTHER cell beats it by more than the margin, consistently,
     for `time_to_trigger` consecutive samples.

Ping-pong counting logic is unchanged conceptually (a handover back to
a recently-left cell, within a short time window) but now "recently
left cell" means any of the N-1 other cells, not just "the other one".
"""

import numpy as np

HYSTERESIS_MARGIN_DB = 1.0
TIME_TO_TRIGGER_SAMPLES = 2
PING_PONG_WINDOW_S = 10.0


def naive_multicell_handover(noisy_rsrp):
    """noisy_rsrp: shape (n_steps, n_cells). Returns serving cell index per step."""
    return np.argmax(noisy_rsrp, axis=1)


def hysteresis_multicell_handover(noisy_rsrp, margin_db=HYSTERESIS_MARGIN_DB,
                                   time_to_trigger=TIME_TO_TRIGGER_SAMPLES):
    n_steps, n_cells = noisy_rsrp.shape
    serving = np.zeros(n_steps, dtype=int)
    serving[0] = np.argmax(noisy_rsrp[0])

    current = serving[0]
    # track how many consecutive samples each candidate cell has been
    # beating the current serving cell by more than the margin
    consecutive_count = np.zeros(n_cells, dtype=int)

    for t in range(1, n_steps):
        rsrp_current = noisy_rsrp[t, current]
        beats_margin = noisy_rsrp[t] > (rsrp_current + margin_db)
        consecutive_count = np.where(beats_margin, consecutive_count + 1, 0)
        consecutive_count[current] = 0  # current cell can't "beat itself"

        triggered = np.where(consecutive_count >= time_to_trigger)[0]
        if len(triggered) > 0:
            # if multiple candidates triggered simultaneously, pick the strongest
            best = triggered[np.argmax(noisy_rsrp[t, triggered])]
            current = best
            consecutive_count[:] = 0

        serving[t] = current

    return serving


def analyze_handovers_multicell(serving, t, window_s=PING_PONG_WINDOW_S):
    """Same ping-pong logic as Scenario 1, generalized to N cells."""
    handover_times = []
    handover_from_to = []

    for i in range(1, len(serving)):
        if serving[i] != serving[i - 1]:
            handover_times.append(t[i])
            handover_from_to.append((serving[i - 1], serving[i]))

    ping_pong_count = 0
    for i in range(1, len(handover_from_to)):
        prev_from, prev_to = handover_from_to[i - 1]
        cur_from, cur_to = handover_from_to[i]
        time_gap = handover_times[i] - handover_times[i - 1]
        if cur_to == prev_from and time_gap <= window_s:
            ping_pong_count += 1

    return {
        "total_handovers": len(handover_times),
        "ping_pongs": ping_pong_count,
        "handover_times": handover_times,
        "handover_from_to": handover_from_to,
    }


if __name__ == "__main__":
    from step4_hexgrid_model import generate_multicell_rsrp

    data = generate_multicell_rsrp()
    t = data["t"]

    print("Ground truth true handovers:",
          int(np.sum(np.diff(data["true_serving_cell"]) != 0)))

    print()
    print("=" * 70)
    print("BASELINE 1: Naive (strongest cell wins, no memory)")
    print("=" * 70)
    sv_naive = naive_multicell_handover(data["noisy_rsrp"])
    r_naive = analyze_handovers_multicell(sv_naive, t)
    print(f"Total handovers : {r_naive['total_handovers']}")
    print(f"Ping-pongs      : {r_naive['ping_pongs']}")
    rate = 100 * r_naive['ping_pongs'] / max(r_naive['total_handovers'], 1)
    print(f"Ping-pong rate  : {rate:.1f}%")

    print()
    print("=" * 70)
    print(f"BASELINE 2: Hysteresis ({HYSTERESIS_MARGIN_DB}dB, {TIME_TO_TRIGGER_SAMPLES} samples TTT)")
    print("=" * 70)
    sv_hyst = hysteresis_multicell_handover(data["noisy_rsrp"])
    r_hyst = analyze_handovers_multicell(sv_hyst, t)
    print(f"Total handovers : {r_hyst['total_handovers']}")
    print(f"Ping-pongs      : {r_hyst['ping_pongs']}")
    rate = 100 * r_hyst['ping_pongs'] / max(r_hyst['total_handovers'], 1)
    print(f"Ping-pong rate  : {rate:.1f}%")
