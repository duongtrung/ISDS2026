# PAPER_GUIDE.md — how the paper is built and how to update it

Source: `paper/main.tex` (Springer LNCS; `llncs.cls` + `splncs04.bst` next to it).
Build: `cd paper && tectonic main.tex` → `main.pdf`
(currently 15 pp). Bib: `paper/refs.bib`.

## The golden rule: numbers are never typed by hand
Prose numbers are LaTeX macros (`\NumAmortJ`, `\NumSpeedup`, …) defined in
`paper/tables/headline.tex` and `paper/tables/ext_headline.tex`, which are **generated**
from `results/*.json`. Tables (`rq1_main.tex`, `rq3_transfer.tex`, `datasets.tex`,
`modality.tex`, `capacity.tex`, `flow_transfer.tex`) and figures (`*.png`) are likewise
generated. To change a number: change the experiment → re-run its generator → recompile.
Pipeline: `results/*.json` → `src/gen_tables_video.py` / `src/gen_ext_tables.py` /
`src/plots_video.py` / `src/plots_ext.py` / `src/dataset_stats.py` → `paper/tables/*` +
`paper/figures/*` → `tectonic`.

## Structure of main.tex (fixed per project brief)
Abstract → §1 Intro (contributions, **Notation** para, RQ1–3) → §2 Related Work →
§3 Experimental Setup (Datasets+Table, **Models** = Architecture/Inference/Training with
Eqs 1–2, **Algorithm 1**, **Table 2 hyperparameters**, Computing, Scenarios S1–S5,
Visualization & Results: RQ1 (+ cost model) / RQ2 / RQ3 + Robustness + S4/S5 +
flow-transfer paras) → §4 Remark & Discussion (why baseline fails, what the energy adds,
**why refinement is inert**, **What to watch for when using EBMs**, Limitations) →
§5 Conclusion (RQ reflection) →
Reproducibility (1 sentence). Tables 1 & 3 use `\uparrow/\downarrow` arrows; the only
competitor is the **Long-run Langevin baseline**; the two "Ours" rows are K=0 and K=5.

## Status: ACCEPTED (camera-ready)
Final title: *Amortized Initialization versus Langevin Refinement for Motion-Guided
Segmentation: A Controlled Study of Conditional Energy-based Models (EBMs)*.
Authors (final): Dung Ngoc Le Ha (CTUET/CTU), Tran Thanh Dien (CTU, corresponding),
Nghia Duong-Trung (DFKI Berlin). Length: **15 pages** (the venue limit) — adding anything
new now requires trimming something else first.

### Remaining placeholder
- Repo URL: `\url{https://github.com/PLACEHOLDER/fast-ebm-motion}` in §Reproducibility —
  replace with the real GitHub URL once the repo is pushed. Also update `CITATION.cff`
  and the BibTeX block in `README.md` with volume/pages/DOI when the proceedings appear.

### Preamble requirements (do not remove)
`orcidlink` (ORCID icons), `marvosym` (+ `\DeclareRobustCommand{\Envelope}{\Letter}` —
marvosym calls the glyph `\Letter`, and it must be robust to survive llncs's `\author`
expansion), `algorithm2e` (Algorithm 1), `multirow` (Table 2).

## How to update the paper for a rerun
1. Re-run experiments (see REPRODUCE.md) → refreshes `results/*.json`.
2. Regenerate assets:
   `python src/gen_tables_video.py; python src/gen_ext_tables.py; python src/plots_video.py;
    python src/plots_ext.py; python src/dataset_stats.py; python src/qualitative.py`
3. `cd paper && tectonic main.tex`. Prose auto-tracks the macros; only edit prose for wording.

## How to add a new experiment / table / figure
- New scenario: add training/eval, write results to a `results/*.json`, add a generator
  (mirror `gen_ext_tables.py`: read JSON → write `paper/tables/<name>.tex` + macros to
  a `*_headline.tex`), `\input` it in main.tex, add a Scenario `S#` and a results paragraph.
- New figure: mirror `plots_ext.py` (matplotlib, save to `paper/figures/<name>.png`), add a
  guarded `\begin{figure}…\IfFileExists{figures/<name>.png}{…}{…}…\end{figure}`.
- New macro: `\providecommand{\NumX}{??}` default in the main.tex preamble, then have the
  generator emit `\renewcommand{\NumX}{…}` into the `*_headline.tex` it writes. **Macro names
  must be letters only (no digits).**

## Optional future extension (NOT needed — the paper is at the 15 pp limit)
MoCA (motion-only camouflage) as a 4th dataset: write `src/moca.py` (mirror
`segtrackv2.py`), precompute flow, add a `--dataset moca` branch to `eval_video.py`, then
zero-shot eval the variants + flow baseline and add a row to `flow_transfer.tex`. The
`--modality`/`--ch` plumbing already supports it. Would require trimming elsewhere to fit.

## Reviews already applied (don't redo)
5-reviewer adversarial correctness pass (fixed the K=0/K=5 speedup mismatch, refinement
honesty, J&F transfer framing) and a 4-lens editorial polish (booktabs, equations, spelling,
arrows). The paper is internally consistent and honest about the negative/mixed results
(refinement inert; flow transfers better than learned). Keep that honesty.
