"""
STEP 1 - Physical model: UE movement + noisy RSRP from two cells
==================================================================

Goal of this step: before touching Viterbi at all, we need realistic-ish
data to feed it. So we simulate:
  1. A UE moving in a straight line at constant speed past two cells.
  2. The TRUE RSRP each cell would produce at each point in time, based on
     a simple path-loss model (signal weakens with distance).
  3. Noisy RSRP - what a real receiver would actually measure, because
     RSRP readings are never clean (fading, interference, etc).

We keep this in plain NumPy so every line is inspectable.
"""

import numpy as np

# ---------------------------------------------------------------------
# 1. Scenario geometry
# ---------------------------------------------------------------------
# Cell A sits at x = 0 meters. Cell B sits at x = 1000 meters.
# The UE starts at x = -200 (well inside Cell A's territory) and moves
# in a straight line to x = 1200 (well inside Cell B's territory).
CELL_A_POS = 0.0
CELL_B_POS = 1000.0

UE_START_X = -200.0
UE_END_X = 1200.0
UE_SPEED_MPS = 18.0         # ~65 km/h, car/urban-train speed - fast enough that the
                             # UE spends less time near the crossover, and hysteresis
                             # has less time to "wait out" the noise

TIME_STEP_S = 1.0           # one RSRP measurement report per second (typical-ish)

# ---------------------------------------------------------------------
# 2. Path-loss model (this is what makes RSRP fall off with distance)
# ---------------------------------------------------------------------
# Real 3GPP path-loss models are more complex (frequency-dependent,
# environment-dependent, etc). For our purposes we use a simplified
# log-distance path loss model, which captures the essential behaviour:
# signal strength drops roughly logarithmically with distance.
#
#   RSRP(d) = P_TX - PATH_LOSS_EXPONENT * 10 * log10(d / d0) - constant
#
# We don't need to be 3GPP-accurate here - we need *realistic shape*:
# strong near the cell, weak far away, smooth falloff.

TX_POWER_DBM = -60.0        # reference RSRP at 1 meter distance (arbitrary but realistic-ish)
PATH_LOSS_EXPONENT = 3.5    # typical urban macro-cell value is 3-4
MIN_DISTANCE_M = 1.0        # avoid log(0) when UE is exactly at the cell


def rsrp_from_distance(distance_m):
    """True (noiseless) RSRP in dBm as a function of distance to the cell."""
    d = np.maximum(distance_m, MIN_DISTANCE_M)
    return TX_POWER_DBM - PATH_LOSS_EXPONENT * 10 * np.log10(d)


# ---------------------------------------------------------------------
# 3. Build the UE trajectory
# ---------------------------------------------------------------------
def build_trajectory():
    total_distance = UE_END_X - UE_START_X
    total_time_s = total_distance / UE_SPEED_MPS
    n_steps = int(total_time_s / TIME_STEP_S)

    t = np.arange(n_steps) * TIME_STEP_S
    x = UE_START_X + UE_SPEED_MPS * t
    return t, x


# ---------------------------------------------------------------------
# 4. Compute true and noisy RSRP from both cells along the trajectory
# ---------------------------------------------------------------------
NOISE_STD_DB = 10.0  # standard deviation of the noise, in dB.
                      # Real fast-fading/shadowing noise is often modeled
                      # as log-normal, i.e. Gaussian in the dB domain -
                      # which is exactly what we're doing here.
                      # 8dB is on the higher end - dense urban, lots of
                      # multipath/interference - deliberately chosen to
                      # stress-test the baseline algorithms.

RNG_SEED = 42  # fixed seed so results are reproducible while we're developing


def generate_rsrp_traces():
    t, x = build_trajectory()

    dist_to_a = np.abs(x - CELL_A_POS)
    dist_to_b = np.abs(x - CELL_B_POS)

    true_rsrp_a = rsrp_from_distance(dist_to_a)
    true_rsrp_b = rsrp_from_distance(dist_to_b)

    rng = np.random.default_rng(RNG_SEED)
    noisy_rsrp_a = true_rsrp_a + rng.normal(0, NOISE_STD_DB, size=t.shape)
    noisy_rsrp_b = true_rsrp_b + rng.normal(0, NOISE_STD_DB, size=t.shape)

    return {
        "t": t,
        "x": x,
        "true_rsrp_a": true_rsrp_a,
        "true_rsrp_b": true_rsrp_b,
        "noisy_rsrp_a": noisy_rsrp_a,
        "noisy_rsrp_b": noisy_rsrp_b,
    }


if __name__ == "__main__":
    data = generate_rsrp_traces()

    print(f"{'t(s)':>6} {'x(m)':>8} {'trueA':>8} {'noisyA':>8} {'trueB':>8} {'noisyB':>8}")
    # print every 20th sample so the output is readable
    for i in range(0, len(data["t"]), 20):
        print(f"{data['t'][i]:6.0f} {data['x'][i]:8.1f} "
              f"{data['true_rsrp_a'][i]:8.2f} {data['noisy_rsrp_a'][i]:8.2f} "
              f"{data['true_rsrp_b'][i]:8.2f} {data['noisy_rsrp_b'][i]:8.2f}")

    print(f"\nTotal samples: {len(data['t'])}")
    print(f"Crossover point (where true RSRP_A == true RSRP_B) should be near x=500m "
          f"since both cells have identical path-loss parameters.")
