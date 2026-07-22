# SAEvo — Stacked Autoencoder Evolution Hypothesis

Simulation code for the paper *"Stacked Autoencoder Evolution Hypothesis"*.

The model is a minimal artificial chemistry of three molecular species,
`M13`, `M7`, and `M3` (sequences of length 13, 7, and 3). Each molecule is
**both** an information-storing state **and** a catalyst: a molecule of one type
takes another molecule as input and produces a molecule of another type via a
convolutional (encoding) or deconvolutional (decoding) transformation. The
catalytic parameters of every molecule are read directly from its own state,
so molecular representations and reaction operators co-evolve.

Running the reaction network under selection for self-replication, the
population spontaneously develops a **hierarchical encode–decode (stacked
autoencoder) structure**: information in `M13` can be compressed down to `M3`
and reconstructed back, as measured by a low reconstruction error.

## Repository structure

```
particle.py            Core simulation (reactions, selection, PCA snapshots,
                       reconstruction-error logging).
run_experiments.sh     Reproduces all runs behind Figures 5–7.
figures/
  plot_fig6.py         Figure 6: reconstruction error over time (single run).
  plot_fig7.py         Figure 7: two length presets vs. fixed-weights control.
requirements.txt
```

## Installation

```bash
python -m venv venv && source venv/bin/activate      # or use conda
pip install -r requirements.txt
```

A CUDA GPU is used automatically if available; otherwise the simulation runs on
CPU (slower).

## Reproducing the results

### 1. Run the simulations

```bash
bash run_experiments.sh
```

If your interpreter is inside a conda environment, pass it explicitly:

```bash
PYTHON="conda run -n <env> python" bash run_experiments.sh
```

This produces, for 5 random seeds each (1000 steps per run):

| Run name                    | Configuration                     | Used in       |
|-----------------------------|-----------------------------------|---------------|
| `13_7_3_seed{1..5}`         | preset `(M13, M7, M3)`            | Fig. 5, 6, 7  |
| `13_5_3_seed{1..5}`         | preset `(M13, M5, M3)`            | Fig. 7        |
| `fixed_weights_seed{1..5}`  | `(M13, M7, M3)`, weights frozen   | Fig. 7        |

Outputs are written to:

- `particle_visualizations/<run_name>/` — per-step PCA snapshots and
  `reconstruction_errors.json`
- `saved_particles/<run_name>/` — population state per step

**Figure 5** (PCA of the molecular populations at selected steps) is produced
directly by `particle.py` as the per-step images in
`particle_visualizations/13_7_3_seed1/`.

### 2. Generate the figures

```bash
python figures/plot_fig6.py   # -> figures/fig6_reconstruction_error.png
python figures/plot_fig7.py   # -> figures/fig7_preset_control_comparison.png
```

## Running a single simulation manually

```bash
python particle.py --seed 1 --run_name my_run --preset 13-7-3 --max_iter 1000
```

Key arguments:

| Argument           | Description                                                    |
|--------------------|----------------------------------------------------------------|
| `--seed`           | Random seed (reproducible).                                    |
| `--run_name`       | Output subdirectory name.                                      |
| `--preset`         | Molecule lengths: `13-7-3` or `13-5-3`.                        |
| `--max_iter`       | Number of simulation steps.                                    |
| `--fixed_weights`  | Control: freeze catalytic parameters (no state→catalyst update).|

## Notes on the reconstruction-error metric

Because a molecule reacts with **every** catalyst in the population, encoding a
molecule does not yield a single canonical intermediate but many candidate
products. The reconstruction error is therefore the **nearest-neighbor** mean
squared error: for each original molecule, the minimum MSE over its decoded
candidates against the original population of the same type (Eq. 1 in the
paper).

## Citation

If you use this code, please cite the accompanying paper. Code:
https://github.com/HiroIizuka/SAEvo
