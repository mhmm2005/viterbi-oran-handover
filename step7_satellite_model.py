"""
STEP 7 - Satellite handover scenario: LEO satellite passes
==============================================================

Key difference from terrestrial (Scenarios 1 & 2)
----------------------------------------------------
On the ground, a UE's mobility is essentially unpredictable to the
network ahead of time. But a satellite's position is NOT unpredictable
- orbital mechanics are deterministic. A network (or ground terminal)
knows, in advance, exactly when a given satellite will rise above the
horizon, when it will peak, and when it will set. This means we can,
in principle, build a MUCH more informed transition prior for Viterbi
than "handovers are just generically rare" - we can use the known
schedule directly. We'll build the physical model first here, then
explore that idea in step8.

Physical model (simplified but representative)
--------------------------------------------------
We simulate a sequence of N satellite passes overhead. Each pass has
an elevation profile: it rises from the horizon (0 degrees), climbs to
some peak elevation, then sets back to the horizon, over a period of a
few minutes - this mimics a real LEO pass (e.g. Starlink-like orbits
have passes lasting roughly 4-10 minutes).

  - Signal strength increases with elevation (higher elevation = more
    directly overhead = shorter slant range + less atmosphere to
    punch through).
  - Noise/fade INCREASES at low elevation (more atmosphere, more rain
    fade, more multipath near the horizon) - this is different from
    our terrestrial scenarios where we used constant noise. This is
    the "signal is inherently less reliable low on the horizon" effect
    that's a defining feature of satellite links.

Passes are scheduled to overlap partially, so a new satellite rises
before the previous one sets - this overlap window is exactly where
the handover decision has to be made, and it's the region we care about.
"""

import numpy as np

# ---------------------------------------------------------------------
# Pass schedule: each satellite's (rise_time, peak_time, set_time, peak_elevation)
# ---------------------------------------------------------------------
# We hand-design a schedule of 4 overlapping passes across ~20 minutes,
# each with a different peak elevation (some passes go nearly overhead,
# some are lower/grazing passes - both happen in reality depending on
# the satellite's ground track relative to the observer).

PASS_SCHEDULE = [
    # (rise_time_s, peak_time_s, set_time_s, peak_elevation_deg)
    (0,    150,  320,  75),   # Sat 0: high overhead pass
    (260,  420,  600,  35),   # Sat 1: lower/grazing pass, overlaps Sat 0's tail
    (560,  740,  920,  60),   # Sat 2: medium-high pass, overlaps Sat 1's tail
    (860, 1020, 1200,  45),   # Sat 3: medium pass, overlaps Sat 2's tail
]
N_SATS = len(PASS_SCHEDULE)
TOTAL_DURATION_S = 1200
TIME_STEP_S = 1.0

# ---------------------------------------------------------------------
# Signal model
# ---------------------------------------------------------------------
RSRP_AT_ZENITH_DBM = -90.0     # signal strength if a satellite were directly overhead (el=90)
DB_LOSS_PER_DEGREE = 0.45      # extra attenuation per degree below zenith (simplified linear model)

NOISE_BASE_DB = 3.0            # noise floor even at high elevation (receiver noise, minor scintillation)
NOISE_EXTRA_AT_HORIZON_DB = 9.0  # additional noise/fade that fades in as elevation drops toward 0


def elevation_profile(t, rise, peak, set_, peak_el):
    """
    Simple parabolic arc: 0 at rise and set times, peak_el at peak time.
    Returns elevation in degrees (0 if outside the pass window).
    """
    el = np.zeros_like(t, dtype=float)
    in_view = (t >= rise) & (t <= set_)

    # Use two separate parabolic halves (rise->peak, peak->set) so the
    # arc doesn't have to be symmetric in time (real passes often aren't).
    rising = in_view & (t <= peak)
    setting = in_view & (t > peak)

    if peak > rise:
        frac = (t[rising] - rise) / (peak - rise)
        el[rising] = peak_el * (1 - (1 - frac) ** 2)  # eased rise
    if set_ > peak:
        frac = (t[setting] - peak) / (set_ - peak)
        el[setting] = peak_el * (1 - frac ** 2)  # eased fall

    return el


def rsrp_from_elevation(el_deg):
    """True (noiseless) RSRP as a function of elevation. -inf-like when not visible."""
    rsrp = RSRP_AT_ZENITH_DBM - DB_LOSS_PER_DEGREE * (90 - el_deg)
    rsrp = np.where(el_deg > 0, rsrp, -999.0)  # effectively invisible below horizon
    return rsrp


def noise_std_from_elevation(el_deg):
    """Noise standard deviation increases as elevation drops toward the horizon."""
    el_clipped = np.clip(el_deg, 0, 90)
    return NOISE_BASE_DB + NOISE_EXTRA_AT_HORIZON_DB * (1 - el_clipped / 90)


def generate_satellite_pass_data(seed=0):
    t = np.arange(0, TOTAL_DURATION_S, TIME_STEP_S)
    n = len(t)

    elevations = np.zeros((n, N_SATS))
    for i, (rise, peak, set_, peak_el) in enumerate(PASS_SCHEDULE):
        elevations[:, i] = elevation_profile(t, rise, peak, set_, peak_el)

    true_rsrp = rsrp_from_elevation(elevations)
    noise_std = noise_std_from_elevation(elevations)

    rng = np.random.default_rng(seed)
    noisy_rsrp = true_rsrp + rng.normal(0, 1, size=true_rsrp.shape) * noise_std

    # ground truth: whichever visible satellite has the strongest TRUE rsrp
    true_serving_sat = np.argmax(true_rsrp, axis=1)

    return {
        "t": t,
        "elevations": elevations,
        "true_rsrp": true_rsrp,
        "noisy_rsrp": noisy_rsrp,
        "noise_std": noise_std,
        "true_serving_sat": true_serving_sat,
    }


if __name__ == "__main__":
    data = generate_satellite_pass_data()
    t = data["t"]

    n_true_handovers = int(np.sum(np.diff(data["true_serving_sat"]) != 0))
    print(f"Total duration: {TOTAL_DURATION_S}s, {N_SATS} satellite passes scheduled")
    print(f"Ground-truth true handovers: {n_true_handovers}")

    changes = np.where(np.diff(data["true_serving_sat"]) != 0)[0]
    for c in changes:
        print(f"  t={t[c+1]:6.0f}s: Sat {data['true_serving_sat'][c]} -> "
              f"Sat {data['true_serving_sat'][c+1]}  "
              f"(elevations at that moment: {[f'{e:.0f}' for e in data['elevations'][c+1]]})")

    print(f"\nNoise std range across the run: "
          f"{data['noise_std'][data['elevations']>0].min():.1f} - "
          f"{data['noise_std'][data['elevations']>0].max():.1f} dB "
          f"(higher = lower elevation = noisier)")
