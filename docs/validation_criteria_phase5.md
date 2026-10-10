# Validation criteria and analysis plan, fifth phase

This document fixes the settings and criteria of the fifth-phase analyses
before any of them is run. It was written on 2026-10-10, after a fourth round
of critique of the draft article, when every result of the first four phases
was known (docs/validation_criteria_phase4.md and its results commit 016046d).
The fourth phase was written and run within about 28 hours of the first-phase
results; this phase follows within a day. Nothing below changes a threshold or
a criterion of an earlier phase: F1 adds Monte Carlo replicates of a fixed
procedure, F2 reads the stored criteria differently for data without a second
sensor, F3 repeats existing analyses on the corrected body scale and F4
repeats the second-sensor comparison with matched projection, frame rate and
duration. As before, a commit in a repository the authors control is not an
independent time stamp, and the code was written with the same AI tool as that
of the earlier phases. No run time was measured on any split listed here.

## F1. Monte Carlo precision of the calibration

Twenty splits per cell leave wide intervals: 0 of 20 is compatible with a true
rate of up to 16.8% (exact 95% bound).

* Further splits j = 20, ..., 99 (80 per cell, 100 in all) for the cells
  pc_g1 (no planted difference) at n = 20, 30 and 45; pc_g4 (the pre-specified
  gain) at n = 45; N1 (image) and N2 (framing) at n = 45. N3 (video-only
  artifact) is assembled from pc_g4 and pc_g1 at n = 45 as in E4, so it also
  has 100 splits. The other cells keep their 20 splits.
* Partitions: n = 45, `pseudo_partition(j)` of the third phase (seed
  PSEUDO_SEED + j); n = 20 and 30, the E3 rule (seed 20261301 + 1000 n + j).
* Classifiers, features, criteria (i)-(v) and the graded outcome are those of
  E3-E5, unchanged.
* Reported for every cell: the number of splits meeting each criterion,
  concluding movement (A) and reaching A or B, each with an exact
  (Clopper-Pearson) 95% interval. For cells with 100 splits the first 20 are
  the fourth-phase splits.
* Criteria (unchanged): specificity, A in at most 10% of the splits at g = 1
  (every n) and under each of N1-N3; graded outcome, A or B in at most 10% of
  the same cells; the D1 sensitivity criterion (A in at least 80% at g = 4,
  n = 45). **Added (P5):** a criterion is reported as *met with its interval*
  only if the whole exact 95% interval lies on the right side of its
  threshold (upper bound at most 10%; lower bound at least 80%); otherwise as
  met (or not met) on the point estimate only.
* The intervals describe split-to-split and Monte Carlo variability for these
  45 children. They do not include the variability between cohorts, which one
  cohort cannot estimate; the article says so.

## F2. Outcome without a second sensor

The context of use is a single smartphone video, but criterion (v) needs a
second recording channel, so with one channel neither A nor B can be reached.

* **Single-channel outcome (P5):** with criterion (v) unavailable, the outcome
  is *S, no recording explanation found, movement not corroborated* if (i)-(iv)
  are met, and *C, not attributable to movement* otherwise. A and B are not
  reachable.
* Computed from the stored criteria of every calibration split (criterion (v)
  ignored) and reported with exact intervals for every cell.
* No pass criterion: S is a screening result, not evidence of movement.
  Expected before the evaluation: under N3 the video-side criteria are those
  of pc_g4 on the same splits, so S occurs under N3 exactly as often as under
  pc_g4 at n = 45. The article states what this means: with one channel, the
  protocol can rule a movement interpretation out but cannot rule it in.

## F3. The analyses of what the models use, on the corrected scale

The per-frame shoulder distance inflates the high-frequency content of the
prepared coordinates about four-fold (third phase, D7), which degrades the
velocity and acceleration channels. The analyses behind "the models rely
mostly on posture" are repeated with the 1-s running median of the shoulder
distance as the scale (`normalise_scale_variant(..., "shoulder_median")`, as
in D7).

* **PACE-median:** the test-case network (arm PACE, configuration, frozen
  partition, seeds and three repetitions unchanged; checkpoints saved) trained
  and evaluated on the corrected-scale sequences (`run_cv.py --set
  data.scale=shoulder_median`).
* **Mean-pose-median:** the mean-pose classifier on the corrected-scale
  sequences, same nested cross-validation (three repetitions). Gait-LR-median
  and Gait-RF-median exist (D7) and are reported with them.
* **Frame order:** every PACE-median checkpoint scores its test children with
  their detected frames permuted (five permutations, seeds as in D9 of the
  third phase: 20261103 + k + 1000 x child index).
* **Channel zeroing:** the position, velocity and acceleration columns set to
  0 at the encoder input of every PACE-median checkpoint, AUC per
  (repetition, inner model) as before.
* AUCs with 2,000-resample stratified bootstrap intervals over children.
* **Wording rule (P5)**, applied to the corrected-scale results: the models
  *use temporal order* if the frame-shuffle drop exceeds 0.05 with a lower
  95% bound above 0; they *use velocity or acceleration* if zeroing either
  channel lowers the network's AUC by more than 0.05; *posture suffices* if
  Mean-pose-median is within 0.05 of the best corrected-scale movement model
  (PACE-median, Gait-LR-median, Gait-RF-median). The article says the models
  rely *mostly on posture* only if posture suffices and neither of the other
  two holds; *on movement over time* if either of the other two holds and
  posture does not suffice; and *on both* otherwise.

## F4. Second-sensor comparison matched for projection, frame rate and duration

The third-phase comparison set video descriptors of the whole clip (median
about 2 s) against descriptors of the depth skeleton (about 1 s) computed in
three dimensions by different code.

* **Projection-matched depth descriptors:** the depth joints are mapped to the
  landmarks the descriptors use (Head to the nose landmark; shoulders,
  elbows, wrists, hips, knees and ankles, left and right as labeled),
  projected onto the image plane of a camera facing the child (x = -X/Z,
  y = -Y/Z), resampled to 60 frames per second by linear interpolation on the
  recorded time stamps, and prepared as the video landmarks (hip midpoint
  subtracted, divided by the per-frame shoulder distance); the 26 descriptors
  are then computed by the same code as for the video (src/gait_features.py).
* **Duration-matched video descriptors:** the time alignment of the two
  recordings is unknown. For every child, each descriptor is computed in
  every window of the detected video span whose length equals the child's
  depth recording (step 10 frames) and summarized by its median over windows.
  The whole-clip descriptors (as before) are reported alongside.
* **Sample:** every child with both recordings (95); correlations are also
  reported within each group (50 autistic, 45 typically developing), where
  group differences cannot produce them.
* **Reliability:** split-half reliability (Spearman-Brown) of the
  projection-matched depth descriptors (first and second half of the depth
  recording) and of the duration-matched video descriptors (windows in the
  first and in the second half of the detected span).
* **Rule (thresholds of the third phase, unchanged):** a robust video
  descriptor (the nine of D7) agrees with the depth camera if Spearman rho
  between its duration-matched video value and its projection-matched depth
  value is at least 0.3 over all 95 children; a lower correlation counts as
  disagreement only if the reliability ceiling sqrt(r_video r_depth) is at
  least 0.5. This comparison is the primary one and replaces the third-phase
  comparison in the article, which reports the earlier one alongside.
* The ceiling bounds the effect of random error only; differences between
  the channels that are systematic (viewpoint, timing, landmark definitions)
  are not bounded by it. The article therefore calls a disagreement *not
  corroborated*, not evidence that the video descriptor is wrong.

## F5. Reporting decisions (no analysis)

* The blinded two-annotator annotation of E7 was not carried out. The article
  reports the visual annotation as what it is, a first annotation made with
  the help of an AI tool without blinding, uses it descriptively only, and
  draws no criterion from it; the approach index (E7), which needs no
  annotator, is the evidence on task.
* The planted image, framing and video-only differences are described as
  sanity checks of the measured controls and of criterion (v), not as
  evidence of protection against recording differences that no control
  measures; in the positive control, criterion (v) is met by construction
  because the difference is planted in both channels.
* The recording audit (C1) becomes Tier 0 of the protocol, with a stop rule
  written after its results were seen (marked *after* in the criteria table):
  if group is confounded with setting, session, task or frame format and no
  subset of recordings shares them, no movement claim is made from the data;
  the later tiers are then descriptive.
