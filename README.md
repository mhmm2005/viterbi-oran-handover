# Viterbi/HMM O-RAN Handover Project — Package Contents

## What's in here

- **viterbi-handover-article.md** — the full write-up. Open with any
  Markdown viewer (VS Code, Typora, GitHub, Obsidian, etc.) so the
  charts render inline. Keep this file in the same folder as `assets/`
  or the images won't display.
- **assets/** — all 10 charts referenced in the article (PNG, generated
  with matplotlib directly from the simulation code, not hand-drawn).
- **code/** — every simulation script, from scratch, NumPy only:
  - `step1_rsrp_model.py` → `step3_viterbi.py`: Scenario 1 (simple two-cell crossing)
  - `step4_hexgrid_model.py` → `step6_multicell_viterbi.py`: Scenario 2 (hex-grid city)
  - `step7_satellite_model.py` → `step9_informed_transition.py`: Scenario 3 (satellite) + informed-prior extension
  - `scenario_a_stationary.py` → `scenario_e_load_based.py`: the five real-world stress tests (Section 9 of the article)

## Running the code

Each script is self-contained and runnable on its own:

```bash
pip install numpy matplotlib
python3 code/step1_rsrp_model.py
python3 code/step3_viterbi.py
python3 code/scenario_a_stationary.py
# ...etc
```

Later scripts import from earlier ones in the same folder (e.g.
`step3_viterbi.py` imports from `step1_rsrp_model.py`), so run them
from inside the `code/` directory.

## Regenerating the charts

The charts in `assets/` were produced with short matplotlib scripts
built on top of these modules (not included as separate files here,
since they were one-off plotting snippets) — every number in them
traces back to the actual simulation output, not hand-entered data.
