"""
STEP 8 - Viterbi for satellite handover, with elevation-aware confidence
============================================================================

Same core Viterbi machinery as step6 (N-state HMM), with one meaningful
upgrade that's specific to the satellite case:

Because satellite elevation over time is KNOWN in advance (orbital
mechanics, not estimated from noisy measurements), we can also predict
the EXPECTED NOISE LEVEL of each satellite's signal at each moment -
low elevation = more atmosphere = more fade = less trustworthy reading.
This is information a terrestrial UE mobility model doesn't have
(we don't know in advance exactly how a car will move), but a satellite
ground station genuinely does have.

So instead of a single fixed noise_std for all states (as in step6),
we use a PER-STATE, PER-TIME noise estimate derived from elevation:
readings from a satellite currently low on the horizon are trusted less
than readings from a satellite high overhead - which is exactly the
physically correct thing to do.

We also enforce a hard constraint: a satellite below the horizon
(elevation <= 0) cannot be the true serving satellite - this uses
the same "known ephemeris" information.
"""

import numpy as np

P_SWITCH = 0.02


def viterbi_decode_satellite(noisy_rsrp, elevations, noise_std_fn, p_switch=P_SWITCH):
    """
    noisy_rsrp:  shape (n_steps, n_sats)
    elevations:  shape (n_steps, n_sats) - known in advance from orbital predictions
    noise_std_fn: function mapping elevation (deg) -> expected noise std (dB)
    """
    n_steps, n_sats = noisy_rsrp.shape

    expected_noise_std = noise_std_fn(elevations)
    log_emit = noisy_rsrp / (expected_noise_std ** 2)

    # Hard constraint: a satellite below the horizon can't be serving.
    # We enforce this with a very large negative log-probability rather
    # than literal -inf, to keep the arithmetic well-behaved.
    below_horizon = elevations <= 0
    log_emit = np.where(below_horizon, -1e6, log_emit)

    stay_logp = np.log(1 - p_switch)
    switch_logp = np.log(p_switch / (n_sats - 1))
    log_trans = np.full((n_sats, n_sats), switch_logp)
    np.fill_diagonal(log_trans, stay_logp)

    delta = np.zeros((n_steps, n_sats))
    psi = np.zeros((n_steps, n_sats), dtype=int)

    delta[0] = np.log(1.0 / n_sats) + log_emit[0]

    for t in range(1, n_steps):
        scores = delta[t - 1][:, None] + log_trans
        best_prev = np.argmax(scores, axis=0)
        delta[t] = scores[best_prev, np.arange(n_sats)] + log_emit[t]
        psi[t] = best_prev

    states = np.zeros(n_steps, dtype=int)
    states[-1] = np.argmax(delta[-1])
    for t in range(n_steps - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states


if __name__ == "__main__":
    from step7_satellite_model import generate_satellite_pass_data, noise_std_from_elevation
    from step5_multicell_baseline import (
        naive_multicell_handover, hysteresis_multicell_handover,
        analyze_handovers_multicell
    )

    data = generate_satellite_pass_data()
    t = data["t"]
    n_true = int(np.sum(np.diff(data["true_serving_sat"]) != 0))

    sv_naive = naive_multicell_handover(data["noisy_rsrp"])
    r_naive = analyze_handovers_multicell(sv_naive, t)

    sv_hyst = hysteresis_multicell_handover(data["noisy_rsrp"])
    r_hyst = analyze_handovers_multicell(sv_hyst, t)

    sv_viterbi = viterbi_decode_satellite(data["noisy_rsrp"], data["elevations"],
                                           noise_std_from_elevation)
    r_viterbi = analyze_handovers_multicell(sv_viterbi, t)

    print(f"Ground truth true handovers: {n_true}\n")
    print(f"{'Algorithm':<15}{'Total HO':>10}{'Ping-pongs':>12}")
    for name, r in [("Naive", r_naive), ("Hysteresis", r_hyst), ("Viterbi", r_viterbi)]:
        print(f"{name:<15}{r['total_handovers']:>10}{r['ping_pongs']:>12}")

    print("\nViterbi handover events:")
    for time_val, (frm, to) in zip(r_viterbi["handover_times"], r_viterbi["handover_from_to"]):
        print(f"  t={time_val:6.1f}s   Sat {frm} -> Sat {to}")
