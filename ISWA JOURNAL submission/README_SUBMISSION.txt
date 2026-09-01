ISWA JOURNAL SUBMISSION — FILE CHECKLIST
=========================================

Manuscript title:
  PACE-ASD: Pose-Aware Contiguous Event Saliency-Gated Transformer for
  Autism Spectrum Disorder Screening from Markerless Video

Corresponding author:
  Sireesha Puppala
  Email: sireesha@neuroparadigm.in
  Tel: +91 9490064325

Journal: Information Sciences (ISWA) / Elsevier

Submission date: 2026-08-31

─────────────────────────────────────────────────────────────────────────────
FILES IN THIS FOLDER
─────────────────────────────────────────────────────────────────────────────

  manuscript.tex
      Main LaTeX manuscript (Elsevier CAS single-column, cas-sc.cls template).
      Do NOT submit as PDF — submit the .tex source.

  references.bib
      BibTeX bibliography. Required by manuscript.tex (compile together).

  supplementary_material.tex
      Supplementary material (LaTeX). Contains:
        S1: Full reproducibility guide (one command per result)
        S2: Preprocessing specification
        S3: Per-seed test results for A1 (full table) and summary for A2-A4
        S4: Complete pairwise statistical comparison table
        S5: Architecture parameter counts
        S6: Dataset audit summary
        S7: Platt calibration analysis
        S8: Clip-length distribution

  highlights.txt
      Article highlights — submitted as a SEPARATE FILE in Editorial Manager
      with the word "highlights" in the filename, as required by Elsevier.

  compute_stats.py
      Python script that computes all statistical results reported in the
      manuscript directly from the per-seed JSON files.
      Requires: numpy, scipy. Run from d:/PACE-ASD/ directory.

─────────────────────────────────────────────────────────────────────────────
TEMPLATE FILES (copy to submission directory before compilation)
─────────────────────────────────────────────────────────────────────────────

  From iswa_template_extracted/els-cas-templates/:
    cas-sc.cls          — document class
    cas-common.sty      — common style
    cas-model2-names.bst — bibliography style

─────────────────────────────────────────────────────────────────────────────
COMPILATION INSTRUCTIONS
─────────────────────────────────────────────────────────────────────────────

  1. Copy cas-sc.cls, cas-common.sty, cas-model2-names.bst from template.
  2. Copy result JSON files to same directory (or adjust paths in compute_stats.py).
  3. Compile manuscript:
       pdflatex manuscript.tex
       bibtex manuscript
       pdflatex manuscript.tex
       pdflatex manuscript.tex
  4. Compile supplementary:
       pdflatex supplementary_material.tex

─────────────────────────────────────────────────────────────────────────────
SUBMISSION SYSTEM NOTES (Editorial Manager)
─────────────────────────────────────────────────────────────────────────────

  • Upload manuscript.tex as the main document source (not the PDF).
  • Upload references.bib as "LaTeX Source File".
  • Upload cas-sc.cls, cas-common.sty, cas-model2-names.bst as "LaTeX Source Files".
  • Upload highlights.txt as a separate item with "highlights" in the filename.
  • Upload supplementary_material.tex (and compiled PDF) as "Supplementary Material".
  • Data statement: dataset available at https://doi.org/10.5061/dryad.s7h44j150
  • Code/data statement: GitHub repository (DOI to be assigned on acceptance).

─────────────────────────────────────────────────────────────────────────────
INTEGRITY STATEMENT
─────────────────────────────────────────────────────────────────────────────

  All numerical results in the manuscript are derived directly from:
    results/A1_per_seed.json  (A1, 20 seeds)
    results/A2_per_seed.json  (A2, 20 seeds)
    results/A3_per_seed.json  (A3, 20 seeds)
    results/A4_per_seed.json  (A4, 20 seeds)
    results/A5_mtcformer_per_seed.json  (MTC-Former baseline, 20 seeds)
    results/ablation_results.csv  (all model means/SDs)
    results/supplement_results.csv  (severe-ASD supplement)
    results/interpretability_metrics_{A1,A2,A3}.json

  No results were hand-edited or reported without a traceable source file.
  The compute_stats.py script in this folder reproduces all statistical tests.
  Per-seed freeze date: 2026-08-31 (no further test-set-informed changes).
