"""
STEP 3 - Viterbi decoding for handover, implemented from scratch
===================================================================

Model setup
-----------
Hidden states: S in {0 = "truly served by Cell A", 1 = "truly served by Cell B"}

Observation at each time step: instead of using the two raw noisy RSRP
values directly, we reduce them to a single scalar that's easier to
reason about:

    diff[t] = noisy_rsrp_b[t] - noisy_rsrp_a[t]

  - diff very negative  -> Cell A looks much stronger
  - diff very positive  -> Cell B looks much stronger
  - diff near zero      -> ambiguous (this is where ping-pong happens)

Emission model P(diff[t] | S[t] = state)
-----------------------------------------
We model diff[t] as Gaussian:
  - If truly served by A: diff[t] ~ N(-MARGIN_DB, sigma)
  - If truly served by B: diff[t] ~ N(+MARGIN_DB, sigma)

MARGIN_DB represents "how much stronger the serving cell typically is,
on average, while actually serving a UE" - a reasonable RF-planning
assumption, not cheating with ground-truth position.

sigma = sqrt(2) * NOISE_STD_DB, because diff is the difference of two
independent noisy measurements, and variances add when you subtract
independent random variables.

Transition model P(S[t] | S[t-1])
-----------------------------------
  P(stay in same state)   = 1 - P_SWITCH
  P(switch to other state) = P_SWITCH

P_SWITCH is small, reflecting that a UE only "truly" changes serving
cell rarely relative to the 1-second reporting interval - this is
exactly the prior knowledge that lets Viterbi ignore single noisy
blips instead of reacting to them.

Algorithm
---------
Standard Viterbi, done in log-space for numerical stability (otherwise
probabilities underflow to 0 after multiplying dozens of small numbers
together).
"""

import numpy as np
from step1_rsrp_model import generate_rsrp_traces, NOISE_STD_DB
from step2_threshold_handover import analyze_handovers

# ---------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------
MARGIN_DB = 5.0          # assumed typical dominance of the serving cell, in dB
P_SWITCH = 0.02          # prior probability of truly switching cell per sample
                          # (~1/50 -> expected dwell time of ~50 samples/seconds,
                          # reasonable for a ~78-second crossing with one real handover)

SIGMA_DIFF = np.sqrt(2) * NOISE_STD_DB  # combined noise std of the diff signal


def log_gaussian_pdf(x, mean, sigma):
    """Log of the Gaussian PDF - avoids underflow vs. computing the PDF directly."""
    return -0.5 * np.log(2 * np.pi * sigma ** 2) - ((x - mean) ** 2) / (2 * sigma ** 2)


def viterbi_decode(diff, margin_db=MARGIN_DB, p_switch=P_SWITCH, sigma=SIGMA_DIFF):
    """
    Run Viterbi decoding on the diff signal.
    Returns the most likely sequence of hidden states (0 = A, 1 = B).
    """
    n = len(diff)
    n_states = 2  # 0 = A, 1 = B

    # Emission means: state 0 (A) expects negative diff, state 1 (B) expects positive diff
    means = np.array([-margin_db, margin_db])

    # log-transition matrix
    log_trans = np.array([
        [np.log(1 - p_switch), np.log(p_switch)],
        [np.log(p_switch), np.log(1 - p_switch)],
    ])

    # log-emission matrix: log_emit[t, s] = log P(diff[t] | state=s)
    log_emit = np.zeros((n, n_states))
    for s in range(n_states):
        log_emit[:, s] = log_gaussian_pdf(diff, means[s], sigma)

    # delta[t, s] = log-probability of the best path ending in state s at time t
    delta = np.zeros((n, n_states))
    psi = np.zeros((n, n_states), dtype=int)  # backpointers

    # Initialization: uninformative prior over starting state (0.5 / 0.5)
    delta[0] = np.log(0.5) + log_emit[0]

    # Forward pass
    for t in range(1, n):
        for s in range(n_states):
            # for each candidate previous state s_prev, compute score of
            # transitioning into s, then take the best one
            scores = delta[t - 1] + log_trans[:, s]
            best_prev = np.argmax(scores)
            delta[t, s] = scores[best_prev] + log_emit[t, s]
            psi[t, s] = best_prev

    # Backward pass: reconstruct the most likely path
    states = np.zeros(n, dtype=int)
    states[-1] = np.argmax(delta[-1])
    for t in range(n - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states


if __name__ == "__main__":
    data = generate_rsrp_traces()
    t = data["t"]
    diff = data["noisy_rsrp_b"] - data["noisy_rsrp_a"]

    serving_viterbi = viterbi_decode(diff)
    result_viterbi = analyze_handovers(serving_viterbi, t)

    print("=" * 70)
    print(f"VITERBI (margin={MARGIN_DB}dB, p_switch={P_SWITCH})")
    print("=" * 70)
    print(f"Total handovers : {result_viterbi['total_handovers']}")
    print(f"Ping-pongs      : {result_viterbi['ping_pongs']}")
    rate = 100 * result_viterbi['ping_pongs'] / max(result_viterbi['total_handovers'], 1)
    print(f"Ping-pong rate  : {rate:.1f}%")

    print()
    print("Handover events (Viterbi):")
    for time_val, (frm, to) in zip(result_viterbi["handover_times"],
                                    result_viterbi["handover_from_to"]):
        print(f"  t={time_val:6.1f}s   Cell {frm} -> Cell {to}")
