"""
STEP 6 - N-state Viterbi for the hex-grid multi-cell scenario
=================================================================

Generalizing from the 2-state case (Scenario 1) to N=7 states.

Hidden states: S in {0, 1, ..., 6} - which cell is truly serving the UE.

Emission model P(noisy_rsrp[t] | S[t] = c)
---------------------------------------------
In Scenario 1 we modeled the emission using the SIGN and MAGNITUDE of
the difference between two cells' readings. With N cells that pairwise
approach doesn't generalize cleanly, so instead we use a simpler and
more general idea:

  log P(observation | S[t] = c)  is proportional to  noisy_rsrp[t, c] / NOISE_STD_DB^2

In words: the stronger cell c's own reading is, the more likely it is
that cell c is truly serving the UE right now. This drops the additive
constant terms of the full Gaussian log-pdf (they don't depend on which
state we're comparing, so they don't affect which state wins) and
keeps just the part that matters: the residual term ~ RSRP / sigma^2.
This generalizes the 2-cell "diff" idea to any number of cells and
requires no per-pair tuning.

Transition model P(S[t] | S[t-1])
-------------------------------------
  P(stay in same cell)        = 1 - P_SWITCH
  P(switch to any OTHER cell) = P_SWITCH / (N_CELLS - 1)   [uniform over the rest]

Same idea as Scenario 1: switching cell is rare, so a single noisy
sample favoring a different cell isn't enough to flip the decoded
path - Viterbi needs sustained evidence.
"""

import numpy as np

P_SWITCH = 0.02  # same prior as Scenario 1: small probability of a true handover per sample


def viterbi_decode_multicell(noisy_rsrp, noise_std_db, p_switch=P_SWITCH):
    """
    noisy_rsrp: shape (n_steps, n_cells)
    Returns the most likely sequence of serving-cell indices, shape (n_steps,).
    """
    n_steps, n_cells = noisy_rsrp.shape

    # log-emission: proportional score per cell, per time step
    log_emit = noisy_rsrp / (noise_std_db ** 2)

    # log-transition matrix: diagonal = stay, off-diagonal = switch (uniform)
    stay_logp = np.log(1 - p_switch)
    switch_logp = np.log(p_switch / (n_cells - 1))
    log_trans = np.full((n_cells, n_cells), switch_logp)
    np.fill_diagonal(log_trans, stay_logp)

    delta = np.zeros((n_steps, n_cells))
    psi = np.zeros((n_steps, n_cells), dtype=int)

    # uninformative uniform prior over starting cell
    delta[0] = np.log(1.0 / n_cells) + log_emit[0]

    for t in range(1, n_steps):
        # scores[s_prev, s] = delta[t-1, s_prev] + log_trans[s_prev, s]
        scores = delta[t - 1][:, None] + log_trans  # shape (n_cells, n_cells)
        best_prev = np.argmax(scores, axis=0)       # best previous state for each candidate s
        delta[t] = scores[best_prev, np.arange(n_cells)] + log_emit[t]
        psi[t] = best_prev

    states = np.zeros(n_steps, dtype=int)
    states[-1] = np.argmax(delta[-1])
    for t in range(n_steps - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states


if __name__ == "__main__":
    from step4_hexgrid_model import generate_multicell_rsrp, NOISE_STD_DB
    from step5_multicell_baseline import (
        naive_multicell_handover, hysteresis_multicell_handover,
        analyze_handovers_multicell
    )

    data = generate_multicell_rsrp()
    t = data["t"]
    n_true_handovers = int(np.sum(np.diff(data["true_serving_cell"]) != 0))

    sv_naive = naive_multicell_handover(data["noisy_rsrp"])
    r_naive = analyze_handovers_multicell(sv_naive, t)

    sv_hyst = hysteresis_multicell_handover(data["noisy_rsrp"])
    r_hyst = analyze_handovers_multicell(sv_hyst, t)

    sv_viterbi = viterbi_decode_multicell(data["noisy_rsrp"], NOISE_STD_DB)
    r_viterbi = analyze_handovers_multicell(sv_viterbi, t)

    print(f"Ground truth true handovers: {n_true_handovers}\n")

    print(f"{'Algorithm':<15}{'Total HO':>10}{'Ping-pongs':>12}{'Excess vs truth':>18}")
    for name, r in [("Naive", r_naive), ("Hysteresis", r_hyst), ("Viterbi", r_viterbi)]:
        excess = r["total_handovers"] - n_true_handovers
        print(f"{name:<15}{r['total_handovers']:>10}{r['ping_pongs']:>12}{excess:>18}")

    print("\nViterbi handover events:")
    for time_val, (frm, to) in zip(r_viterbi["handover_times"], r_viterbi["handover_from_to"]):
        print(f"  t={time_val:6.1f}s   Cell {frm} -> Cell {to}")
