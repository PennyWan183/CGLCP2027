# CGLCP: Core Simulation Experiments

This repository contains the code for the **main simulation experiments of CGLCP**, including graph construction, conformal calibration methods, experiment runners, and figure generation. It is intended to help researchers reproduce the core simulation results and explore the roles of local contamination and unsafe-score placement.

**The code for the real-data experiments will be released upon formal acceptance of the paper.** The current release focuses on the two core simulation experiments described below; it does not include the real-data pipeline or the full set of supplementary experiments.

## Experiments

| Experiment | Research question | Figure |
|---|---|---|
| Local necessity | Is a global contamination rate sufficient when contamination varies across local graph neighborhoods? | `local_necessity.pdf` |
| Unsafe-score robustness | How does the placement of unsafe calibration scores affect coverage and interval width? | `score_robustness.pdf` |

Both are controlled, score-level simulations on explicit graphs. They do not require biological data, a trained prediction model, a GPU, or a LaTeX installation.

## Getting started

Use Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The checked environment uses Python 3.10.18, NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3, and Matplotlib 3.10.0. Package versions are pinned in `requirements.txt`.

### 1. Reproduce the figures from the reported results

```bash
python plot_figures.py
```

This reads the included `frozen/` summary tables and produces both figures as PDF and PNG files in `outputs/figures/`. It is the fastest way to inspect the reported results and requires no simulation rerun.

### 2. Check the full pipeline

```bash
python reproduce.py --profile smoke
```

This runs both experiments with three repetitions per setting and saves per-replicate data, summaries, configurations, and figures under `outputs/smoke/`. These small runs test the implementation; their estimates are not the paper results.

### 3. Rerun the core simulations with the paper settings

```bash
python reproduce.py --profile paper
```

Results are saved under `outputs/paper/`:

```text
outputs/paper/
├── local_necessity/
│   ├── replicates.csv
│   ├── summary.csv
│   └── config.json
├── score_robustness/
│   ├── replicates.csv
│   ├── summary.csv
│   └── config.json
├── figures/
│   ├── local_necessity.pdf
│   ├── local_necessity.png
│   ├── score_robustness.pdf
│   └── score_robustness.png
└── run_metadata.json
```

To rerun only one experiment:

```bash
python reproduce.py --profile paper --only local
python reproduce.py --profile paper --only robustness
```

Use `--output-dir PATH` to choose another output directory. Runs do not overwrite the included frozen summaries. Full simulation runs, particularly GraphLCP calibration, are more expensive than plotting; runtime depends on CPU and numerical libraries.

To plot results from a completed rerun:

```bash
python plot_figures.py \
  --local-summary outputs/paper/local_necessity/summary.csv \
  --robustness-summary outputs/paper/score_robustness/summary.csv \
  --output-dir outputs/paper/figures
```

## Experimental settings

### Local necessity

The working graph has a fixed global false-safe fraction of 0.10. The experiment changes where causal edges are missing, producing different contamination levels around a query while preserving the global fraction. The local pool and unsafe membership are derived from graph distances and directed reachability.

- Local pool size: 60.
- Local unsafe counts: 1, 3, 6, 12, 15, and 24.
- Repetitions: 800 per setting; random seed: `1024`.
- Nominal miscoverage: `alpha=0.10`; certificate failure budget: `rho=0.02`.
- Verification: a uniform sample of 30 local members.
- GraphLCP bandwidth: 4.
- Safe calibration and test scores: `abs(N(0,1))`.
- Unsafe calibration scores: `abs(N(0,0.1^2))`.

The seven methods are Pooled CP, Global-rate plug-in, GraphLCP, PCS Corrected, Localized PCS Corrected, CGLCP, and Oracle local (rho). Pooled CP and GraphLCP use all calibration interventions. The PCS selector is fixed to the working-graph predicted-safe set; its localized version restricts that set to the query neighborhood. These choices isolate pooling, localization, and contamination correction.

### Unsafe-score robustness

This experiment fixes the local pool and varies the distribution of unsafe scores.

- Local pool size: 100; unsafe count: 20; verification size: 40.
- Repetitions: 4,000 per regime; original random seed: `1024`.
- `alpha=0.10`, `rho=0.02`.
- Safe calibration and test scores: `abs(N(0,1))`.

| Regime | Unsafe score distribution |
|---|---|
| Benign | `abs(N(0,1))` |
| Moderate | `abs(N(0,0.5^2))` |
| Severe | `abs(N(0,0.1^2))` |
| Adversarial | Exactly zero |

The six methods are Local CP, Global-rate plug-in, CGLCP, Oracle local (alpha), Oracle local (rho), and Oracle-safe CP. This comparator set isolates the effect of unsafe-score placement and the cost of imperfect contamination information; it serves a different purpose from the seven-method local-necessity comparison.

### Random-number reproducibility

The original robustness experiment used a generator shared with a preceding simulation (S1). To recover that stream, the paper-profile runner first executes S1 with 3,000 repetitions per each of seven settings, then runs the robustness experiment (S3). S1 outputs are discarded. The intervening S2 calculation is deterministic and consumes no random numbers, so it is omitted.

Starting S3 directly with the same seed does **not** recover the reported draws. `reproduce.py` handles this sequencing automatically. Smoke mode shortens both stages and intentionally produces different draws.

## Methods and metrics

CGLCP draws a uniform verification sample, forms a hypergeometric upper bound on the number of unsafe local members, and applies the contamination-aware conformal rank. The implementation is in `cglcp_methods.py`.

The three oracle benchmarks have different information:

- **Oracle local (rho)** knows the true local unsafe count but retains CGLCP's adjusted conformal budget, isolating the cost of estimating contamination.
- **Oracle local (alpha)** knows the count and uses the full nominal miscoverage budget.
- **Oracle-safe CP** knows every membership label, removes unsafe scores, and applies ordinary CP to the remaining safe scores.

GraphLCP uses graph-distance kernel weights and a localized conformal acceptance-level adjustment, including the test-point mass. It is not an unadjusted weighted empirical quantile. Its source attribution is retained in `baselines/graph_lcp.py`.

Coverage includes infinite intervals as covered. Finite rate measures the fraction of finite outputs. Conditional coverage and median finite width use only finite outputs. Simulated intervals are centered at zero, so finite width is twice the selected score threshold.

The local-necessity plot shows saved 95% Wilson intervals for proportion metrics; finite-width medians have no error bars. The robustness plot shows point estimates without error bars. All 24 method-by-regime finite rates in the frozen robustness summary equal one, so its finite-rate panel is omitted. New runs are not forced to satisfy this property.

Red horizontal lines indicate nominal coverage 0.90; the gray vertical line in local necessity marks the global contamination fraction 0.10. CGLCP is orange, and its legend entry is last. The plotting script does not change the underlying estimates.

## Repository structure

```text
reproduce.py                   # paper-scale and smoke experiment entry point
plot_figures.py                 # publication figures from summary CSVs
plot_style.py                   # consistent labels, colors, and markers
cglcp_methods.py                # certificates and conformal interval methods
simulation/
  graph_design.py               # explicit reference and working graphs
  local_necessity.py            # main local-contamination experiment
  score_robustness.py           # main score-placement experiment and RNG warm-up
baselines/
  graph_lcp.py                  # localized conformal baseline
frozen/
  local_necessity/              # reported summary and configuration
  score_robustness/             # reported summary and original suite configuration
requirements.txt
VALIDATION.json                 # recorded implementation checks
```

The frozen robustness configuration records the original suite for provenance; this release's entry point runs only the two core experiments and the necessary RNG warm-up.

## Reproducibility checks

The release has been checked by plotting both frozen summaries and running both experiments end to end in smoke mode from outside the repository directory. The core robustness routines retain the original numerical function bodies. Reorganizing the code does not change the random-number sequencing or method implementations. Paper-scale simulations were not rerun as part of this repository cleanup.

Small differences in rendered PDF bytes can arise from timestamps, fonts, or backend versions. Compare numerical summaries rather than PDF file hashes when assessing experimental reproducibility.

## Acknowledgment

The GraphLCP calibration implementation follows the original localized conformal implementation at [LeyingGuan/LCP](https://github.com/LeyingGuan/LCP). See its source header for attribution.
