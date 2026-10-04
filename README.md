# CPCI: Simulation-Calibrated Conformal Prediction for Counterfactual Group-Contrast Effects in Parkinson's Disease MRI

Code for *Simulation-Calibrated Conformal Prediction for Counterfactual
Group-Contrast Effects in Parkinson's Disease MRI: Valid Intervals without
Counterfactual Ground Truth* (Tao, Li, Guo).

## Contents

| File | Role |
|---|---|
| `rd_cci_v8_cpci.py` | Full pipeline: ROI feature extraction, twin-VAE generator, CPCI-Sim calibration (with the Proposition-1 floor), all experiments |
| `make_figs_and_tables_v2.py` | Fig. tiers (subject-cluster bootstrap CIs), Fig. deploy, tab:balance LaTeX rows |
| `make_revision_figs.py` | frontier / cdf_comparison / stress2_heatmap / effect_regions figures |
| `make_rate_fig.py` | Fig. rate (reads rate_curve.csv; no re-run needed) |
| `make_table2_ppmi.py` | Table 2 (tab:acq) PPMI demographics from the analyzed 754-subject subset |

## Experiments (paper section -> command)

```
python rd_cci_v8_cpci.py main        # Sec 4.2 three-tier results (20 splits)
python rd_cci_v8_cpci.py abl         # Sec 4.2 ablation without training ITEs
python rd_cci_v8_cpci.py deploy --target NEUROCON --m-list 5 10 15 20 25 30 --repeats 5   # Sec 5 Fig. deploy
python rd_cci_v8_cpci.py deploy --target TaoWu    --m-list 5 10 15 20 25 30 --repeats 5
python rd_cci_v8_cpci.py balance --n-target 42 --repeats 20                                 # Sec 4.2 / Appendix C
python rd_cci_v8_cpci.py rate        # Sec 4.8 rate/latent sweep (~3 h)
python rd_cci_v8_cpci.py stress      # Sec 4.7 severity-family misspecification
python rd_cci_v8_cpci.py stress2     # Sec 4.7 effect-direction mismatch
python rd_cci_v8_cpci.py stress3     # Sec 4.7 independent effect mechanism
python rd_cci_v8_cpci.py sens        # Sec 4.6 calibration hyperparameter grid
python rd_cci_v8_cpci.py shift       # Sec 4.1 domain-shift statistics
python rd_cci_v8_cpci.py cdf         # Fig. cdf input (5 splits)
```

All commands accept `--data-root` (default `M:/MRI`) and `--out-root`
(default `M:/MRI/Results_RDCCI_v7`); reuse previously extracted features
with `--feat-root <dir>`. All seeds are fixed inside the script
(20 splits, seeds 42-61).

## Data

PPMI (https://www.ppmi-info.org), NEUROCON and TaoWu via the 1000
Functional Connectomes Project
(https://fcon_1000.projects.nitrc.org/indi/retro/parkinsons.html).
Expected layout and atlas paths are documented in the module docstring of
`rd_cci_v8_cpci.py`.

## License

MIT.
