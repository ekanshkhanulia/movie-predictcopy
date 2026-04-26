================================================================
REPORT DRAFTS — index and how to use these files
================================================================

This folder contains one txt per section that you (Ekansh) need to
write yourself. Sections 1, 2.1, and 2.2 are not here because your
teammate is writing those.

The four sections below have already been numerically reconciled
against the final run (K = 1, post-Chapter-10 honest baseline).
The K = 100 attempt is documented as a tested-and-reverted result
in Section 4.3 and as a separate row in Section 3 Table 3.

Files in this folder
--------------------

  2.3_training_and_hyperparameters.txt
       FULL prose for "2.3 Training & Hyperparameters", written
       in strict chronological order so the reader can follow the
       project as a story:
         Para 1 -> baseline at start of project (the configuration
                   used by the grid search and both experiment
                   phases)
         Para 2 -> the three-stage hyperparameter search performed
                   on top of that baseline
         Para 3 -> the post-search optimiser refinement (Adam
                   beta_2 + weight_decay)
         Para 4 -> the post-search negative-sampling refinement
                   (history-exclusion alignment, plus a brief
                   forward reference to the K = 100 attempt)
         Para 5 -> final schedule, reproducibility, full
                   cumulative final-config recap

  3_results_and_comparisons.txt
       Tables and short factual prose. Contains:
         Table 1 - final val/test metrics (filled in)
         Table 2 - top-3 grid configurations (rank-3 still <FILL>)
         Table 3 - training-side ablation (filled in, including a
                   final "tested-and-reverted: K = 100" row)
         Optional Table 4 - maxlen sensitivity from 2nd-exp-phase

  4_observations_and_improvements.txt
       FULL prose for "4 Observations & Improvements".
         4.1  architecture is not the bottleneck
         4.2  negative results: LR warmup, sqrt(d) scaling
         4.3  diagnosed plateau, unsuccessful K = 100 attempt
         4.4  future work: hard negatives, BPR, padding mask,
              cosine LR

  5_conclusion_and_improvements.txt
       Short summary (~paragraph or two). One full version and one
       one-paragraph compressed alternative for tight pages.

Remaining things to fill in before submission
---------------------------------------------

  1. 3_results_and_comparisons.txt, Table 2: rank-3 entry. Pull
     from results/ablation_results.csv. Section 4.1's "all three
     of the top configurations use 128 hidden units and maxlen
     50" claim assumes whichever rank-3 entry you pick still has
     hidden = 128 and maxlen = 50; this is true for the two
     candidates listed in the NUMBERS TO PASTE IN block.
  2. 3_results_and_comparisons.txt, Table 3: the baseline val
     Recall@10 number marked <FILL: 0.0524>. CHANGE_LOG only has
     three decimals; pull the exact value from
     results/ablation_results.csv (run_id 43).
  3. Sanity-check the citation [2] index against the teammate's
     reference list - if the teammate gave SASRec a different
     index, edit the citations across all four files in one pass.

What to copy into the .tex file
-------------------------------

In each txt, ONLY the part marked PROSE (and the explicit
\begin{table} blocks in section 3) is the actual report content.
The bullet lists labelled "WHAT THIS SECTION DELIBERATELY LEAVES
OUT", "BEFORE SUBMITTING - DOUBLE-CHECK", "NUMBERS TO PASTE IN",
"NUMBERS TO FILL IN", and so on are notes for you - do NOT paste
those into the .tex file.

Source-of-truth mapping
-----------------------

  If a section needs ...                Look in ...
  -------------------------------       --------------------------
  data preprocessing decisions          data.py + teammate's 2.1
  model architecture decisions          model.py + teammate's 2.2
  grid search top 3 + stage-1 setup     1expermentphase(...).txt
                                        + results/ablation_results.csv
  dropout / lr stage-2 results          1expermentphase(...).txt
  maxlen=200 / epoch=100 stage-3        2ndexpphase(...) folder
  Adam beta + weight decay (kept)       CHANGE_LOG Chapter 8
  LR warmup (reverted, neg result)      CHANGE_LOG Chapter 9
  negative sampler alignment (kept)     CHANGE_LOG Chapter 10
  sqrt(d) scaling (reverted)            CHANGE_LOG Chapter 11
  K=1 vs K=100 multi-negative
       (tested, reverted)               CHANGE_LOG Chapter 12

If anything in the drafts disagrees with what you ran, trust the
terminal output and edit the prose - the prose was written from
the files above and the values may need a refresh.
