# Validation criteria and analysis plan, fourth phase

This document fixes the settings and pass/fail criteria of the fourth-phase
analyses before any of them is run. It was written on 2026-10-09, after the
third review of the article, when every result of the first three phases was
known: in particular the positive control of the third phase (movement
concluded in 12 of 20 splits at g = 4, below the 80% required; criteria (iii)
and (iv) limited it), the incremental-value results of D3, the time-of-day
AUCs of the linear encoding (0.40 over all children, 0.26 within the cropped
videos) and the AI-assisted visual annotation of the 95 recordings. The run
time of the calibration pipeline was measured on one split that is not part of
this plan (pseudo-group split 99 at g = 2), whose AUCs were not printed or
inspected. The code of this phase was written with the same AI tool as that of
the second and third phases. As before, a commit in a repository the authors
control is not an independent time stamp; this document and its commit are
the record, and the article says so.

## E1. Time of day as a non-linear control (C3)

The linear logistic regression on the hour of the depth recording (D4) cannot
represent a group recorded both earlier and later in the day than the other,
which is the case here; it gave out-of-fold AUCs below 0.5.

* New encoding: the hour enters as a cubic B-spline basis (scikit-learn
  `SplineTransformer`, 5 knots spread evenly over the range of the training
  children, degree 3, constant extrapolation), fitted within each training
  set, followed by the same standardization and L2-penalized logistic
  regression with C chosen in the inner folds (grid 0.001-10).
* Applied wherever the time of day enters: *Time-of-day*, *Depth-recording*
  (its time-of-day column) and *All-controls* (and *All-controls-noformat*,
  D12). The linear versions are kept under the names *-linear* and reported in
  the Supplementary Material.
* Re-run under the frozen partition (three repetitions); D11 and D12 re-run;
  label-permutation null (D2 settings, 200 permutations) for *Time-of-day* and
  *All-controls*.
* Criteria: the interval convention and every criterion that uses these
  controls are applied unchanged to the new rows.

## E2. Incremental value by nested stacking (C3, replaces the test of D3)

D3 fitted the second-level model on the out-of-fold scores of all children,
so that the scores used to fit it for one child had been produced by models
trained on that child's label, and it used a weak ridge penalty in place of a
penalty that keeps the estimates finite under separation; its likelihood-ratio
test is invalid when the groups are (nearly) separable. The new test:

* Level 1: out-of-fold probabilities (as logits) of the controls *Scene*,
  *Time-of-day* (E1), *Frame-format* and *Demographics*, and of one movement
  score.
* Cross-fitting: for every repetition r and outer fold k, the second-level
  models are fitted on the outer-training children and applied to the
  outer-test children of that fold. The outer-training children enter with
  inner-fold scores: for the feature classifiers, the classifier with the
  hyperparameters selected for fold k is refitted on each inner training set
  and scores that inner fold's validation children; for the test-case network,
  each child is scored by the stored inner model whose validation set
  contained it (with its recalibration). The outer-test children enter with
  their stored out-of-fold scores. No child's label influences its own
  second-level prediction.
* Level 2: Firth-penalized logistic regression (Firth, 1993) of the label on
  the standardized level-1 logits, (a) controls only and (b) controls and the
  movement score.
* Statistics, averaged over repetitions: cross-fitted AUC of (a) and (b) and
  the AUC gain (b - a); cross-fitted mean log loss of (a) and (b) and the
  log-loss reduction (a - b); 95% percentile intervals from 2,000 stratified
  bootstrap resamples of children, paired, with the out-of-fold predictions
  fixed (seed 20261405).
* **Criterion (P4):** a movement score adds information beyond the controls if
  the intervals of both the AUC gain and the log-loss reduction lie above 0.
* Real data: movement scores of the test-case network, Gait-RF and Depth-RF;
  all children, and within the cropped videos (second-level models fitted and
  evaluated on the cropped children only); three repetitions. The D3 results
  are reported alongside as the third-phase specification.

## E3. Operating characteristics: the protocol over gain and sample size

* Pseudo-groups of the 45 typically developing children. n = 45: the 20 splits
  and partitions of D1 (pseudo-groups of 22 and 23). n = 30 and n = 20: for
  split j = 0..19, a random permutation (numpy default_rng(20261301 + 1000 n +
  j)) of the 45 children; the first n/2 form pseudo-group 1 and the next n/2
  pseudo-group 0; one repetition of stratified five-fold outer and three-fold
  inner cross-validation with the same seed.
* Planted difference of D1 (video landmarks and depth skeletons) with gain
  g in {1, 2, 3, 4, 6}; g = 1 is the null.
* Classifiers per split: Gait-LR, Gait-RF, Mean-pose (movement);
  Frame-format, Scene, Time-of-day (E1), Demographics, Tracking-quality,
  All-controls (E1) (controls); Depth-LR, Depth-RF, Depth-recording (E1)
  (depth). The test-case network is not run (it was never the best movement
  model in the third-phase positive control).
* Criteria per split:
  (i) every control has an AUC below 0.70;
  (ii) the best movement model (highest AUC of Gait-LR, Gait-RF, Mean-pose) has
  a 95% bootstrap lower bound above 0.5 and exceeds All-controls with a paired
  bootstrap interval above 0 (2,000 resamples);
  (iii) E2 with the best movement model's score (one repetition);
  (iv) at least one of the 26 descriptors is a robust difference (group effect
  adjusted for age and sex with q < 0.05 over the 26 descriptors, 500
  bootstrap resamples; same sign with a 95% interval excluding 0 for smoothed
  trajectories and for the first 90 frames); the arm-swing-specific version
  of D1 is reported alongside;
  (v) the better depth classifier (Depth-LR or Depth-RF) has a lower bound
  above 0.5 and an AUC above Depth-recording (E1).
  Movement is concluded if (i)-(v) are all met.
* Reported for every (n, g): the share of splits meeting each criterion and
  concluding movement, the graded outcome (E5) and the mean AUCs.
* The cells n = 45, g in {1, 2, 4} re-evaluate the third-phase positive
  control with E1 and E2 (same splits); D1's criterion (movement in at least
  80% of splits at g = 4 and at most 10% at g = 1) is applied to them and
  reported next to the pre-specified third-phase result, which stands.

## E4. Planted recording differences (negative controls)

The pseudo-group splits of D1 (n = 45) with no planted movement and one of
three recording differences in pseudo-group 1:

* **N1, image.** Every frame of the color video is darkened and its contrast
  lowered (every channel multiplied by 0.75) and its saturation raised (HSV
  saturation multiplied by 1.4), as the autistic children's cropped videos
  differed from the typically developing children's. MediaPipe Pose is run on
  the transformed frames (settings of src/preprocess.py); the landmarks,
  validity pattern, visibility and the image statistics of the scene control
  come from the transformed frames; the bit rate is that of the original file.
* **N2, framing.** Every frame is cropped to its central 80% in width and
  height (the frame size changes); landmarks and image statistics as in N1.
* **N3, a movement-like difference in the video only.** The arm-movement gain
  g = 4 in the video landmarks of pseudo-group 1, the depth skeletons
  unchanged. The video-side classifiers are those of (n = 45, g = 4) and the
  depth-side classifiers those of (n = 45, g = 1), which share pseudo-groups,
  partitions and seeds; no new model is trained.
* Criteria (i)-(v) and the graded outcome as in E3; the AUCs of the movement
  models show whether the recording difference reached them.
* **Criterion (P4):** the protocol attributes a planted recording difference
  to recording if it concludes movement in at most 10% of the splits of each
  of N1, N2 and N3.

## E5. Graded outcome

The all-or-nothing rule of D1 was conservative. A graded outcome, defined
after the third-phase positive control (which motivated it) and before any
fourth-phase run:

* **A, attributable to movement:** (i)-(v) all met.
* **B, movement supported but not established:** (i), (ii) and (v) met; (iii)
  or (iv) not met.
* **C, not attributable to movement:** (i), (ii) or (v) not met.

(i), (ii) and (v) guard against recording differences that the controls
measure and against differences confined to the video; (iii) and (iv) limited
the third-phase positive control. Because the motivating splits (n = 45,
g = 4) are re-used, sensitivity there is not an independent test; its
specificity is.

* **Criterion (P4):** grade A or B in at most 10% of the splits at g = 1
  (every n) and under each of N1, N2 and N3. Sensitivity (share of A and of A
  or B over the grid) is reported descriptively.

## E6. Real data

* E1 and E2 applied to the 95 children (all children; cropped videos); the
  criteria table is updated with the new time-of-day rows and the E2 test;
  the D3 results move to the Supplementary Material.
* The graded outcome of E5 is not applied to the real data, whose criteria
  are those of Table 4 (C1-C4); the conclusion is unchanged if any of (i),
  (ii) or (v) fails there.

## E7. Visual audit: blinded human annotation and an objective cross-check

The visual annotation of the 95 recordings (third phase) was made with an AI
tool and was not blinded. It is repeated as follows (codebook:
docs/annotation_codebook.md; material: scripts/visual_audit_phase4.py --kit).

* **Objective cross-check (computed now, before any human annotation):** the
  approach index of each recording, the log ratio of the child's apparent
  size (distance between the shoulder and hip midpoints in pixels, median over
  the last quarter of the detected frames) to that over the first quarter.
  Walking towards the camera enlarges the child in the frame. A recording is
  *approaching* if the index is at least log(1.2). Reported: AUC of the index
  for the annotation "walks towards the camera" and Cohen's kappa between
  *approaching* and that annotation (AI-assisted annotation now; human
  consensus when available), with 2,000-resample bootstrap intervals.
* **Human annotation:** two authors annotate independently. Material: one
  sheet per recording with eight pose overlays evenly spaced over the detected
  frames and the first detected frame without overlay, identified only by a
  random code (numpy default_rng(20261401)), in code order; a contact sheet of
  the first frames for grouping recordings into settings. Child identifiers,
  group labels, file names and frame sizes are not shown; the key is kept by
  an author who does not annotate. Items: walks towards the camera for most of
  the clip; another person visible; gross tracking error (the skeleton on
  another person or grossly misplaced in at least one overlay); setting
  (recordings judged to be in the same place share a label). Annotators do not
  see the AI-assisted annotation or each other's work; afterwards they agree a
  consensus, which replaces the AI-assisted annotation in every analysis.
* Reported: Cohen's kappa with bootstrap intervals for the three binary items;
  adjusted Rand index of the two setting partitions; for each annotator,
  whether any setting contains recordings of both groups; agreement of the
  consensus with the AI-assisted annotation.
* **Criterion (P4):** an item is adequately reliable if kappa is at least
  0.60; otherwise the conclusions that rest on it are reported as uncertain.

## E8. Wording (no analysis)

Criteria are described as "written before the analysis was run" with the
time stamps' limits, not as independent pre-registration; the computational
checks of the analysis code are called cross-checks, because they were written
with the same AI tool as the code they check.
