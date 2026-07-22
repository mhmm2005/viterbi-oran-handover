"""
STEP 4 - Scenario 2: hexagonal cell grid + realistic UE movement
====================================================================

This scenario is a big step up in realism from Scenario 1:

  - Multiple cells (7, in a classic 1-center + 6-neighbor hex layout)
    instead of just 2. This means our hidden state space grows from
    {A, B} to {Cell0, Cell1, ..., Cell6} - a genuinely multi-state HMM.

  - The UE follows a realistic composite trajectory:
      1. A smooth random walk (gentle heading changes, not sharp turns)
         so the path naturally weaves near multiple cell boundaries.
      2. Variable speed - alternating "cruising" and "slow/stopped"
         phases, similar to urban traffic (stop-start driving).

We keep the physical layer (path-loss model, noise model) identical to
Scenario 1 so results are directly comparable in spirit - we're only
changing the geometry and mobility pattern, not the physics.
"""

import numpy as np

# ---------------------------------------------------------------------
# 1. Hexagonal cell grid (1 center cell + 6 surrounding neighbors)
# ---------------------------------------------------------------------
ISD_M = 500.0  # inter-site distance: distance between adjacent cell centers

def build_hex_grid(isd=ISD_M):
    """
    Classic single-ring hexagonal layout: one cell at the origin,
    six neighbors arranged around it at 60-degree intervals, each
    at distance ISD from the center.
    Returns an array of shape (7, 2): [x, y] position of each cell.
    """
    centers = [(0.0, 0.0)]  # Cell 0: the center cell
    for k in range(6):
        angle = np.deg2rad(60 * k)
        x = isd * np.cos(angle)
        y = isd * np.sin(angle)
        centers.append((x, y))
    return np.array(centers)  # shape (7, 2)


CELL_POSITIONS = build_hex_grid()
N_CELLS = len(CELL_POSITIONS)

# ---------------------------------------------------------------------
# 2. Path-loss model (same shape as Scenario 1, reusable constants)
# ---------------------------------------------------------------------
TX_POWER_DBM = -60.0
PATH_LOSS_EXPONENT = 3.5
MIN_DISTANCE_M = 1.0


def rsrp_from_distance(distance_m):
    d = np.maximum(distance_m, MIN_DISTANCE_M)
    return TX_POWER_DBM - PATH_LOSS_EXPONENT * 10 * np.log10(d)


# ---------------------------------------------------------------------
# 3. UE trajectory: smooth random walk + variable speed ("urban traffic")
# ---------------------------------------------------------------------
TIME_STEP_S = 1.0
N_STEPS = 400              # total simulated seconds (~6.5 minutes of driving)
CRUISE_SPEED_MPS = 15.0    # "green light" driving speed (~54 km/h)
SLOW_SPEED_MPS = 2.0       # "traffic/stop" speed
HEADING_NOISE_STD_RAD = 0.25   # how sharply the UE can turn per step (small = smooth)
SPEED_PHASE_MEAN_LEN = 25       # avg number of steps before speed regime switches
RNG_SEED = 7


def build_trajectory(n_steps=N_STEPS, seed=RNG_SEED):
    rng = np.random.default_rng(seed)

    # Start near the center cell, heading in a random direction
    x, y = 0.0, 0.0
    heading = rng.uniform(0, 2 * np.pi)

    xs = np.zeros(n_steps)
    ys = np.zeros(n_steps)
    speeds = np.zeros(n_steps)

    # Build a sequence of "speed phases" (cruise vs slow) using a simple
    # semi-random schedule, to mimic stop-start urban traffic
    is_cruising = True
    steps_left_in_phase = int(rng.exponential(SPEED_PHASE_MEAN_LEN))

    for t in range(n_steps):
        if steps_left_in_phase <= 0:
            is_cruising = not is_cruising
            steps_left_in_phase = max(3, int(rng.exponential(SPEED_PHASE_MEAN_LEN)))
        steps_left_in_phase -= 1

        speed = CRUISE_SPEED_MPS if is_cruising else SLOW_SPEED_MPS
        # small random speed jitter so it's not perfectly binary
        speed *= rng.uniform(0.85, 1.15)

        # smooth random walk: heading drifts gently rather than jumping around
        heading += rng.normal(0, HEADING_NOISE_STD_RAD)

        x += speed * np.cos(heading) * TIME_STEP_S
        y += speed * np.sin(heading) * TIME_STEP_S

        xs[t] = x
        ys[t] = y
        speeds[t] = speed

    t_arr = np.arange(n_steps) * TIME_STEP_S
    return t_arr, xs, ys, speeds


# ---------------------------------------------------------------------
# 4. Generate true + noisy RSRP from all 7 cells along the trajectory
# ---------------------------------------------------------------------
NOISE_STD_DB = 10.0  # same noise level as Scenario 1


def generate_multicell_rsrp(seed=RNG_SEED):
    t, xs, ys, speeds = build_trajectory(seed=seed)
    n = len(t)

    true_rsrp = np.zeros((n, N_CELLS))
    for c in range(N_CELLS):
        dx = xs - CELL_POSITIONS[c, 0]
        dy = ys - CELL_POSITIONS[c, 1]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        true_rsrp[:, c] = rsrp_from_distance(dist)

    rng = np.random.default_rng(seed + 1000)  # different stream from mobility noise
    noisy_rsrp = true_rsrp + rng.normal(0, NOISE_STD_DB, size=true_rsrp.shape)

    # ground-truth "true serving cell" = whichever cell has the strongest
    # TRUE (noiseless) RSRP at each instant - this is our evaluation
    # reference, not something the algorithms get to see
    true_serving_cell = np.argmax(true_rsrp, axis=1)

    return {
        "t": t, "x": xs, "y": ys, "speed": speeds,
        "true_rsrp": true_rsrp, "noisy_rsrp": noisy_rsrp,
        "true_serving_cell": true_serving_cell,
    }


if __name__ == "__main__":
    data = generate_multicell_rsrp()

    print(f"Cell positions (hex grid, ISD={ISD_M}m):")
    for i, (cx, cy) in enumerate(CELL_POSITIONS):
        print(f"  Cell {i}: ({cx:7.1f}, {cy:7.1f})")

    print(f"\nTrajectory: {N_STEPS} steps, {N_STEPS * TIME_STEP_S:.0f}s total")
    print(f"Speed range: {data['speed'].min():.1f} - {data['speed'].max():.1f} m/s")
    print(f"Position range: x=[{data['x'].min():.0f}, {data['x'].max():.0f}], "
          f"y=[{data['y'].min():.0f}, {data['y'].max():.0f}]")

    # How many DISTINCT true-serving-cell segments are there? (i.e. how
    # many times does the ground truth actually change cell)
    changes = np.sum(np.diff(data["true_serving_cell"]) != 0)
    print(f"\nGround-truth true handovers (ideal reference): {changes}")
    print(f"Cells visited (in order of first appearance):")
    seen = []
    for c in data["true_serving_cell"]:
        if not seen or seen[-1] != c:
            seen.append(c)
    print(f"  {seen}")
