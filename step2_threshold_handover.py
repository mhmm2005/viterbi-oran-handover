"""
STEP 2 - Baseline handover algorithms: naive threshold vs hysteresis
=====================================================================

This is the "current state of the art" that Viterbi will be compared
against. Two variants, both operating on ONE noisy RSRP sample at a
time (this is the key weakness we're targeting - no memory of history):

  1. NAIVE: at every time step, connect to whichever cell has the
     higher (noisy) RSRP right now. No margin, no memory.

  2. HYSTERESIS: only switch cells if the candidate cell's RSRP
     exceeds the current serving cell's RSRP by at least a margin
     (HYSTERESIS_MARGIN_DB), and this must hold for a minimum number
     of consecutive samples (TIME_TO_TRIGGER). This mirrors real
     3GPP A3-event style handover triggering (this is genuinely how
     LTE/5G handover works in practice - it's the closest fair
     baseline to compare against).

We also define what counts as a "ping-pong": a handover back to the
cell the UE just left, within PING_PONG_WINDOW_S seconds. This is the
standard definition used in 3GPP mobility robustness studies.
"""

import numpy as np
from step1_rsrp_model import generate_rsrp_traces

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
HYSTERESIS_MARGIN_DB = 1.0     # candidate must beat serving cell by this much
TIME_TO_TRIGGER_SAMPLES = 2    # ...and stay ahead for this many consecutive samples
                                # (chosen via a parameter sweep - see project notes:
                                # a slow/large TTT+margin essentially "solves" ping-pong
                                # on its own but adds handover latency; this setting
                                # reflects a network tuned for responsiveness, which is
                                # where the ping-pong problem actually shows up)
PING_PONG_WINDOW_S = 10.0      # a handover back within this window = ping-pong


def naive_threshold_handover(rsrp_a, rsrp_b):
    """
    At every sample, serve whichever cell is currently stronger.
    Returns an array of serving cell ids: 0 = cell A, 1 = cell B.
    """
    serving = np.where(rsrp_b > rsrp_a, 1, 0)
    return serving


def hysteresis_handover(rsrp_a, rsrp_b, margin_db=HYSTERESIS_MARGIN_DB,
                         time_to_trigger=TIME_TO_TRIGGER_SAMPLES):
    """
    Start on whichever cell is stronger at t=0. Then only switch once the
    OTHER cell has been ahead by more than `margin_db` for `time_to_trigger`
    consecutive samples in a row. This is much closer to real handover
    logic (3GPP A3 event) than the naive version.
    """
    n = len(rsrp_a)
    serving = np.zeros(n, dtype=int)
    serving[0] = 0 if rsrp_a[0] >= rsrp_b[0] else 1

    consecutive_count = 0
    current = serving[0]

    for i in range(1, n):
        candidate = 1 - current  # the "other" cell
        rsrp_current = rsrp_a[i] if current == 0 else rsrp_b[i]
        rsrp_candidate = rsrp_b[i] if current == 0 else rsrp_a[i]

        if rsrp_candidate > rsrp_current + margin_db:
            consecutive_count += 1
            if consecutive_count >= time_to_trigger:
                current = candidate
                consecutive_count = 0
        else:
            consecutive_count = 0

        serving[i] = current

    return serving


# ---------------------------------------------------------------------
# Metrics: count handovers and ping-pongs
# ---------------------------------------------------------------------
def analyze_handovers(serving, t, window_s=PING_PONG_WINDOW_S):
    """
    Walk through the serving-cell sequence and find:
      - every handover event (a change in serving cell)
      - which of those handovers are "ping-pongs": a switch back to the
        cell the UE was on immediately before its previous handover,
        occurring within `window_s` seconds of that previous handover.
    """
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
        # A ping-pong: we go back to the cell we just came from, quickly.
        if cur_to == prev_from and time_gap <= window_s:
            ping_pong_count += 1

    return {
        "total_handovers": len(handover_times),
        "ping_pongs": ping_pong_count,
        "handover_times": handover_times,
        "handover_from_to": handover_from_to,
    }


if __name__ == "__main__":
    data = generate_rsrp_traces()
    t = data["t"]

    print("=" * 70)
    print("BASELINE 1: Naive threshold (strongest signal wins, no memory)")
    print("=" * 70)
    serving_naive = naive_threshold_handover(data["noisy_rsrp_a"], data["noisy_rsrp_b"])
    result_naive = analyze_handovers(serving_naive, t)
    print(f"Total handovers : {result_naive['total_handovers']}")
    print(f"Ping-pongs      : {result_naive['ping_pongs']}")
    print(f"Ping-pong rate  : {100 * result_naive['ping_pongs'] / max(result_naive['total_handovers'],1):.1f}%")

    print()
    print("=" * 70)
    print(f"BASELINE 2: Hysteresis ({HYSTERESIS_MARGIN_DB} dB margin, "
          f"{TIME_TO_TRIGGER_SAMPLES} samples time-to-trigger)")
    print("=" * 70)
    serving_hyst = hysteresis_handover(data["noisy_rsrp_a"], data["noisy_rsrp_b"])
    result_hyst = analyze_handovers(serving_hyst, t)
    print(f"Total handovers : {result_hyst['total_handovers']}")
    print(f"Ping-pongs      : {result_hyst['ping_pongs']}")
    print(f"Ping-pong rate  : {100 * result_hyst['ping_pongs'] / max(result_hyst['total_handovers'],1):.1f}%")

    print()
    print("First 10 handover events (naive):")
    for time_val, (frm, to) in list(zip(result_naive["handover_times"],
                                         result_naive["handover_from_to"]))[:10]:
        print(f"  t={time_val:6.1f}s   Cell {frm} -> Cell {to}")
