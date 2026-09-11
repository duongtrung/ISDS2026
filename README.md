# Amortized Initialization versus Langevin Refinement for Motion-Guided Segmentation

**A Controlled Study of Conditional Energy-based Models (EBMs)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.12](https://img.shields.io/badge/pytorch-2.12%2Bcu130-ee4c2c.svg)](https://pytorch.org/)

Dung Ngoc Le Ha<sup>1,2</sup>, Tran Thanh Dien<sup>2</sup>, Nghia Duong-Trung<sup>3</sup>

<sup>1</sup> Can Tho University of Technology, Can Tho City, Vietnam ·
<sup>2</sup> Can Tho University, Can Tho City, Vietnam ·
<sup>3</sup> German Research Center for Artificial Intelligence (DFKI), Berlin, Germany

> Official code for the LNCS paper. Every number and figure in the paper is
> regenerated from logged JSON by the scripts in this repository — no value is
> ever entered by hand.

---

## What this studies

Conditional EBMs are an appealing formulation for motion segmentation: one learned
energy `E_θ(m | x)` scores how compatible a candidate mask `m` is with a frame's
appearance and motion, and can represent several plausible masks at once. The
obstacle is **sampling** — drawing a mask normally means long-run Langevin MCMC from
noise, which is slow and unstable.

We replace that with a **data-grounded amortized initializer** `q_φ(x)` that proposes a
mask in one forward pass, followed by **K short Langevin refinement steps** under the
same energy. This makes `K` an explicit speed–accuracy dial, and — because `K=0` is
exactly the initializer alone — it lets us **isolate what the learned energy actually
contributes**.

**Headline findings** (full numbers in [RESULTS.md](RESULTS.md)):

| | DAVIS-16 `J` | train (min) | ms/frame |
|---|---|---|---|
| Long-run Langevin (baseline, `N=100`) | 0.074 | 117 | 320.5 |
| **Ours, initializer only (`K=0`)** | **0.227** | 16 | **1.74** |
| **Ours, init + refinement (`K=5`)** | **0.228** | 49 | 18.3 |

- The from-noise baseline **never localizes** (flat `J ≈ 0.074` for `N = 10…400`, even
  after a fair sampler-tuning pass); amortizing the initialization is **184×** faster
  per frame at `K=0` (17.5× at `K=5`) *and* far more accurate.
- **The refinement is largely inert**: `K = 0 → 20` moves region `J` by `+0.002`
  (within noise) while boundary `F` slightly *decreases* — and this still holds at
  4× the parameters. We report this negative result openly.
- **A trivial flow threshold transfers better** zero-shot (SegTrack-v2 `J` 0.355 vs
  0.087), exposing appearance overfitting in all learned models here.

## Quickstart

```bash
# 1. Environment (aarch64 + NVIDIA Blackwell GB10; adjust the CUDA index for your GPU)
uv venv --python 3.12 .venv
UV_HTTP_TIMEOUT=900 uv pip install --python .venv/bin/python \
    torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python matplotlib tqdm
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # -> True

# 2. Datasets + precomputed optical flow (~16 GB, see the script's notes)
bash scripts/download_data.sh

# 3. Train the three variants, then the extension studies
bash scripts/run_all_train.sh     # baseline / initializer / initializer+energy   (~3 h)
bash scripts/run_ext.sh           # S4 modality + S5 capacity                     (~3 h, resumable)

# 4. Evaluate, regenerate every table & figure, compile the paper
bash scripts/run_eval_direct.sh
```

Full step-by-step instructions, including every individual command and the
known pitfalls, are in **[REPRODUCE.md](REPRODUCE.md)**.

## Repository layout

```
src/                    # library + entry points
  models.py               encoder, per-location energy head, amortized initializer
  sampler.py              projected Langevin refinement in mask-logit space
  train_video.py          trains a variant (baseline | feedforward | amortized)
  train_energy.py         fits the energy critic on FROZEN initializer features
  eval_video.py           J / F / J&F + per-frame latency for a step sweep
  tune_sampler.py         fair sampler tuning for the long-run baseline
  davis.py fbms.py segtrackv2.py   dataset loaders (flow-paired)
  precompute_flow.py      RAFT-large optical flow, cached to .npy
  flow_baseline.py        parameter-free Otsu-on-|flow| reference
  metrics.py              region J (IoU) and boundary F
  seed_agg.py             across-seed aggregation (error bars)
  gen_tables_video.py gen_ext_tables.py dataset_stats.py   LaTeX tables + macros
  plots_video.py plots_ext.py qualitative.py               figures
scripts/                # one-command pipelines (see REPRODUCE.md)
configs/                # exact hyperparameters per variant
results/                # logged JSON — the single source of truth for all numbers
runs_video/ runs_*/     # trained checkpoints (small; included for reproducibility)
paper/                  # LaTeX source, generated tables/figures, main.pdf
```

## Documentation

| File | Purpose |
|---|---|
| [REPRODUCE.md](REPRODUCE.md) | Exact commands: environment → data → training → evaluation → PDF |
| [RESULTS.md](RESULTS.md) | Every reported number with its provenance (check reruns against this) |
| [PAPER_GUIDE.md](PAPER_GUIDE.md) | How the paper is built and how to extend it |

## Notes

- **Scale.** The backbones are deliberately small (30K–226K parameters). This is a
  *controlled study of sampling strategy*, not a bid for state-of-the-art accuracy;
  absolute `J` is well below task-specific VOS systems by design.
- **Hardware.** All experiments ran on a single NVIDIA DGX Spark (GB10 Grace-Blackwell,
  20-core Arm CPU, 128 GB unified memory, `aarch64`), PyTorch 2.12.1+cu130.
- **LNCS class files.** `paper/llncs.cls` and `paper/splncs04.bst` are Springer's and
  are *not* redistributed here; download them from the official LNCS author kit and
  place them in `paper/`.

## Citation

TBA

## License

Code released under the [MIT License](LICENSE). Datasets remain under their
original licenses and are not redistributed here.
