# RESULTS.md — final numbers and their provenance

Authoritative values live in `results/*.json` and are rendered into the paper via macros
in `paper/tables/headline.tex` + `paper/tables/ext_headline.tex`. This file is a human
summary so a future rerun can be sanity-checked. Seed 0 unless noted; DAVIS-16 val.
Models are tiny (30–58K params for ch=32) — a sampling-strategy study, not SOTA.

## RQ1 — DAVIS-16 operating points (`results/video_results.json`; Table 1)
| Method | Params | Train (min) | J | F | J&F | ms/frame |
|---|---|---|---|---|---|---|
| Long-run Langevin (baseline), N=100 | 48K | 117 | 0.074 | 0.082 | 0.078 | 320.5 |
| Ours: initializer only (K=0) | 29.5K | 16 | 0.227±0.003 | 0.231 | 0.229 | 1.74 |
| Ours: init + K-step (K=5) | 57.6K | 49 | 0.228±0.002 | 0.230 | 0.229 | 18.3 |

Speedup vs baseline: **184×** at K=0, **17.5×** at K=5 (both = 320.5 / {1.74, 18.3}).
±std are **across 3 seeds** (`results/seed_agg.json`); baseline is single-seed.

## RQ2 — frontier sweep (`results/video_results.json`; Fig. frontier_davis)
- Baseline noise-init: J≈0.073→0.076 for N=10→400 (flat; never localizes).
- Amortized K-sweep: K=0 J=0.227, K=5 0.228, K=10 0.228, K=20 0.229 (F 0.231→0.229, so
  J&F ~flat, best at K=0). Refinement gain over K=0→20 = **+0.002 region J (within noise)**.
- Fair sampler tuning: best = step-size 10, noise 0.005, all configs J≈0.08
  (`results/sampler_tuning.json`).

## RQ3 — zero-shot transfer (`results/video_results.json`; Tables rq3/flow_transfer, Figs)
J&F (region J in parens):
| Method | DAVIS-16 | FBMS-59 | SegTrack-v2 |
|---|---|---|---|
| Long-run Langevin (baseline) | 0.078 (0.074) | 0.115 (0.137) | 0.064 (0.068) |
| Ours (K=0) | 0.229 (0.227) | 0.139 (0.124) | 0.106 (0.086) |
| Ours (K=5) | 0.229 (0.228) | 0.139 (0.124) | 0.106 (0.087) |
- On J&F, Ours beats the baseline on all three. FBMS region-J reverses (baseline 0.137 >
  ours 0.124) — an object-size artifact of the baseline's diffuse masks.

## Flow baseline — trivial Otsu on |flow| (`results/flow_baseline.json`)
Region J: DAVIS **0.109**, FBMS **0.198**, SegTrack **0.355**. Loses on the training
domain but **transfers better than any learned model** zero-shot → learned features
overfit appearance.

## S4 modality — feed-forward, DAVIS (`results/ext_results.json`, ch=32; Table modality)
| Input | J | F | J&F |
|---|---|---|---|
| RGB only | 0.203 | 0.213 | 0.208 |
| Flow only | 0.161 | 0.462 | 0.311 |
| RGB+flow | 0.227 | 0.231 | 0.229 |
RGB drives region J; flow drives boundary F; both is best on J.

## S5 capacity — amortized K-sweep, DAVIS (`results/ext_results.json`; Table/Fig capacity)
| Width | Params | J(K=0) | J(K=20) | ΔK J | F(K=0) | F(K=20) |
|---|---|---|---|---|---|---|
| 32 | 58K | 0.227 | 0.229 | +0.002 | 0.231 | 0.229 |
| 64 | 226K | 0.243 | 0.246 | +0.003 | 0.232 | 0.231 |
Doubling width lifts absolute J (0.227→0.243) but refinement stays marginal (+0.003, F
still drops) → limitation is not just tiny-capacity.

## Marginal cost of one Langevin step (analytical, from the latency sweeps)
Baseline slope (N=10→100): **3.20 ms/step**; ours (K=0→20): **3.25 ms/step**. Both run the
same energy net, so a step costs the same; per-frame cost is `c_enc + S·c_step`
(S ∈ {N,K}) and the speedup is essentially the step-count ratio N/K. Macros
`NumStepBase` / `NumStepOurs`.

## Macro map (prose numbers -> file)
`paper/tables/headline.tex`: NumBaseJ, NumAmortJ, NumFFj, NumSpeedup(184), NumSpeedupKfive(17.5),
NumBaseTrain/NumAmortTrain/NumFFtrain, NumBaseFBMS/NumAmortFBMS/NumBaseSeg/NumAmortSeg,
NumRefineJ(+0.002), NumFkzero/NumFktwenty, NumAmortStd/NumFFStd/NumAmortFBMSstd, NumSeeds(3).
`paper/tables/ext_headline.tex`: NumRgbJ/F, NumFlowJ/F, NumBothJ, NumBigJ(0.243),
NumBigRefine(+0.003), NumFlowDavis/Fbms/Segtrack.
A rerun should reproduce these to ~±0.002 (seed noise). If a number changes, the paper
auto-updates on the next `gen_tables_video.py`/`gen_ext_tables.py` run — never edit by hand.
