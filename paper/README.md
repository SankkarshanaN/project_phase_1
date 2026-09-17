# IEEE paper — compilation notes

`paper.tex` is a self-contained IEEEtran conference paper (`\documentclass[conference]{IEEEtran}`),
built from the measured results in `docs/RESULTS.md` and `CLAUDE.md` as of 2026-09-17.
Figures are copied locally into `figures/` so this directory can be zipped or
uploaded as-is.

## Before you compile

Two placeholders need your actual details — search `paper.tex` for `[Your`:

1. **Institution name, city/country, email addresses** in the `\author` block
   (currently `[Your Institution Name]`, `[City, Country]`, `[email address]`
   for both authors).
2. Check the **acknowledgment** section names your mentor correctly (currently
   generic — "their project mentor").

Nothing else in the file is a placeholder; every number in the results tables
is pulled directly from the project's own measured, committed results.

## Compiling

No LaTeX toolchain was available in the environment this was written in, so
this has been checked structurally (balanced braces/environments, matching
`\cite`/`\bibitem` and `\ref`/`\label` pairs, correct table column counts,
all figure files present) but **not compiled to a PDF**. Two easy options:

- **Overleaf** (recommended, zero setup): create a new project, upload
  `paper.tex` and the `figures/` folder, compile. IEEEtran is preinstalled.
- **Local TeX Live / MiKTeX**: from this directory,
  ```
  pdflatex paper.tex
  pdflatex paper.tex   # run twice for cross-references/citation numbers
  ```

## Structure

- Abstract, keywords
- I. Introduction — motivation, contributions
- II. Related Work — CARLA, YOLO, evidential deep learning (Sensoy et al.
  2018), Kalman/particle filtering, Clark-Evans spatial statistics
- III. System Architecture — the 9-component pipeline, one subsection per
  component group
- IV. Dataset — v2 collection stats, and the 4 defects found/fixed in the
  earlier pilot collection
- V. Experimental Results — evidential classifier, the occlusion
  ground-truth defect + its proof + its fix (two rounds), sensor ablation,
  crossing-intent, adversarial robustness (including the reverted
  radar-clutter fix attempt, reported honestly), auxiliary mechanisms,
  system performance
- VI. Discussion and Limitations — weather API, remaining occlusion miss
  rate, undetectable sensor faults, single-town/simulation-only scope
- VII. Conclusion
- References (7 entries, all real, verifiable citations)

## If you extend this later

The paper deliberately reports two negative/null results in full (the
evidential-uncertainty-coupling null result is summarized but could be
expanded; the reverted radar-clutter baseline fix has its own paragraph) —
keep that register if you add sections rather than trimming them to look
more polished. That honesty is a real, citable methodological contribution
of the underlying project, not padding.
