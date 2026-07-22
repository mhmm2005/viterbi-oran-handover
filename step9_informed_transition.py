"""
STEP 9 - Ephemeris-informed transition prior for satellite Viterbi
========================================================================

In steps 6/8, the transition prior was a CONSTANT: "switching cells is
rare, probability P_SWITCH per sample, always." That's the right
assumption when we genuinely don't know the UE's future path (terrestrial
scenarios 1 & 2).

But for satellites we DO know the future - the pass schedule (rise,
peak, set times of every satellite) is known in advance from orbital
mechanics. So the transition prior doesn't have to be constant: it
should be:

  - Very low probability of switching AT ANY TIME when only one
    satellite is visible (there's no real decision to make - nothing
    to switch TO, so this isn't even really "conservatism", it's just
    correct given the ephemeris)
  - Meaningfully higher probability of switching specifically DURING
    the known overlap windows between consecutive passes - i.e.,
    exactly the windows where a real handover decision might occur.

This means Viterbi's "willingness to believe a handover just happened"
now varies over time in a way that tracks the known schedule, instead
of being a single fixed compromise value. This is the "unique to
satellite" upgrade that terrestrial UE mobility can't get, because we
don't have ephemeris for a car.
"""

import numpy as np


def build_time_varying_p_switch(elevations, p_switch_base=0.001, p_switch_overlap=0.3):
    """
    elevations: shape (n_steps, n_sats), known in advance from orbital predictions.
    Returns p_switch(t): high during multi-satellite-visible overlap windows
    (real handover decision points), very low otherwise (no real decision to make).
    """
    n_visible = np.sum(elevations > 0, axis=1)
    overlap = n_visible >= 2
    p_switch_t = np.where(overlap, p_switch_overlap, p_switch_base)
    return p_switch_t


def viterbi_decode_satellite_informed(noisy_rsrp, elevations, noise_std_fn,
                                       p_switch_base=0.001, p_switch_overlap=0.3):
    """
    Same Viterbi core as step8, but with a TIME-VARYING transition prior
    built from the known pass schedule instead of one constant P_SWITCH.
    """
    n_steps, n_sats = noisy_rsrp.shape

    expected_noise_std = noise_std_fn(elevations)
    log_emit = noisy_rsrp / (expected_noise_std ** 2)
    below_horizon = elevations <= 0
    log_emit = np.where(below_horizon, -1e6, log_emit)

    p_switch_t = build_time_varying_p_switch(elevations, p_switch_base, p_switch_overlap)

    delta = np.zeros((n_steps, n_sats))
    psi = np.zeros((n_steps, n_sats), dtype=int)
    delta[0] = np.log(1.0 / n_sats) + log_emit[0]

    for t in range(1, n_steps):
        p_switch = p_switch_t[t]
        stay_logp = np.log(1 - p_switch)
        switch_logp = np.log(p_switch / (n_sats - 1))
        log_trans = np.full((n_sats, n_sats), switch_logp)
        np.fill_diagonal(log_trans, stay_logp)

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
    from step8_satellite_viterbi import viterbi_decode_satellite
    from step5_multicell_baseline import analyze_handovers_multicell

    data = generate_satellite_pass_data()
    t = data["t"]
    n_true = int(np.sum(np.diff(data["true_serving_sat"]) != 0))

    sv_constant_p = viterbi_decode_satellite(data["noisy_rsrp"], data["elevations"],
                                              noise_std_from_elevation)
    r_constant_p = analyze_handovers_multicell(sv_constant_p, t)

    sv_informed = viterbi_decode_satellite_informed(data["noisy_rsrp"], data["elevations"],
                                                      noise_std_from_elevation)
    r_informed = analyze_handovers_multicell(sv_informed, t)

    print(f"Ground truth true handovers: {n_true}\n")
    print(f"{'Approach':<30}{'Total HO':>10}{'Ping-pongs':>12}")
    print(f"{'Constant P_SWITCH':<30}{r_constant_p['total_handovers']:>10}{r_constant_p['ping_pongs']:>12}")
    print(f"{'Schedule-informed P_SWITCH':<30}{r_informed['total_handovers']:>10}{r_informed['ping_pongs']:>12}")

    print("\nSchedule-informed handover events:")
    for time_val, (frm, to) in zip(r_informed["handover_times"], r_informed["handover_from_to"]):
        print(f"  t={time_val:6.1f}s   Sat {frm} -> Sat {to}")
