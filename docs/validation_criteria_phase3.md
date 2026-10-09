# Validation criteria and analysis plan, third phase

This document fixes the settings and pass/fail criteria of the third-phase
analyses before any of them is run. It is committed to the repository before
the analyses; the commit is the reference for "specified in advance", and the
document and the commit log are to be archived with an independent time stamp
(OSF or Zenodo) before the article is submitted, because a commit in a
repository the authors control is not an independent time stamp.

The first-phase analyses (nested cross-validation, comparators, frame-format,
duration, validity-pattern and demographic controls, the raise event) and the
second-phase analyses (`docs/validation_criteria.md`, commits 4f23692 and
ce6f234: scene, jitter and depth controls, adjusted descriptors, gate
remedies, flap and freeze events, selection references) and all their results
were known when this document was written. Also known: the file modification
times stored in the Dryad archive (D14 below), which were looked at while
the deposit was examined for this plan. Nothing else below has been run.

## Names

The parts of the protocol are renamed so that they no longer collide with the
V3 framework of Goldsack et al. (2020) and with V (valid blocks):

| New | Old | Content | Tier |
|---|---|---|---|
| C1 | (new) | recording audit: metadata, frame format, file and container metadata, time stamps, duplicates, demographics, pose quality control | 1 |
| C2 | V1-V2 | performance under nested cross-validation, simple baselines, label-permutation null | 1 |
| C3 | V3 (controls) | recording, image, tracking-quality, session and demographic controls; subsets; all-controls model; incremental value | 1 |
| C4 | V3 (descriptors, depth) | measurement checks of descriptors: robustness, reliability, second recording channel, what the model uses | 1 |
| C5 | V4-V5 | planted events with reference detectors; selection on real data against untrained networks | 2 |
| C6 | V6 | attribution against untrained networks; integrated gradients; occlusion | 2 |

Tier 1 applies to every model that maps a pose sequence to a group label and
is cheap. Tier 2 applies only when a model outputs segments or attributions.
The protocol is called the movement-or-recording (MOR) protocol.

## Interval convention (applies to all criteria, old and new)

Thresholds apply to point estimates unless a criterion names an interval
bound. When the 95% interval of a quantity crosses the threshold that its
point estimate meets, the outcome is reported as "met on the point estimate
only"; a control is called *uninformative* only if its upper bound is below
0.70, and *informative* if its point estimate is at least 0.70 with a lower
bound above 0.5. A control between these is *not excluded*.

## D1. Semi-synthetic positive control

Purpose: show that the protocol can pass a case, i.e. that its thresholds do
not reject every dataset.

* Children: the 45 typically developing children. For split j = 0..19 a
  random permutation (numpy default_rng(20261101 + j)) assigns the first 22
  children (sorted by subject id, then permuted) to pseudo-group 1 and the
  other 23 to pseudo-group 0.
* Planted difference (arm-swing gain g, pseudo-group 1 only), video: in each
  prepared sequence, for each arm, the position of the wrist, the hand
  landmarks (15-22 of that side) and the elbow relative to the ipsilateral
  shoulder is split into a low-frequency part (15-frame moving average over
  each run of detected frames) and the residual; the deviation of the
  low-frequency part from its mean over the detected frames is multiplied by
  g for the wrist and hand landmarks and by (1 + g) / 2 for the elbow; the
  residual (tracking noise) is unchanged. Depth: the same in the depth-camera
  skeleton (wrist, hand, hand tip and thumb joints by g, elbow by (1 + g)/2,
  relative to the ipsilateral shoulder, 7-frame moving average at about 30
  frames per second).
* Gain rule (no model is trained before it is applied): for g in {1.1, 1.2,
  1.3, 1.5, 2.0}, the AUC of the video arm-swing descriptor alone between the
  pseudo-groups is averaged over the 20 splits; the g whose mean is closest
  to 0.80 is used (ties: the smaller g). The null case g = 1 (no planted
  difference) is always run.
* Everything that is not the pose sequence is kept as it is: frame format,
  image properties, time of day, demographics and depth-recording properties
  of the 45 children are their real values.
* Nested cross-validation per split: one repetition, stratified five-fold
  outer and three-fold inner (seed 20261101 + j), with the fast classifiers
  and controls of C2-C3 (Gait-LR, Gait-RF, Mean-pose, Depth-LR, Depth-RF,
  Frame-format, Scene, Time-of-day, Demographics, Tracking-quality,
  All-controls). The test-case network (arm PACE, unchanged recipe) is
  trained on splits 0-4 at the chosen g and on splits 0-1 at g = 1.
* Protocol outcome per split ("attributable to movement"): (i) every control
  has an AUC below 0.70; (ii) the best movement model has a lower 95% bound
  above 0.5 and exceeds the All-controls model with a paired bootstrap
  interval above 0; (iii) incremental value (D3) met; (iv) arm swing is a
  robust descriptor difference (adjusted for age and sex, smoothed and first
  90 frames, as in the second phase); (v) depth corroboration (second-phase
  rule). Reported: the share of splits in which each of (i)-(v) and all of
  them are met, at the chosen g and at g = 1.
* **Criterion (P):** the protocol passes the positive control if all of
  (i)-(v) are met in at least 80% of the splits at the chosen g, and the
  conclusion "attributable to movement" is reached in at most 10% of the
  splits at g = 1.

## D2. Label-permutation null

* Fast pipeline: for permutation i = 0..199 (rng 20261201 + i), the labels of
  the 95 children are permuted; a new stratified partition (one repetition,
  five outer and three inner folds, seed 20261201 + i) is made for the
  permuted labels and Gait-LR, Scene, Frame-format, Mean-pose and
  All-controls are run exactly as for the real labels; Gait-RF for i < 100.
  AUC over all children and within the cropped videos.
* Network: the test-case network for permutations 0-9 (one repetition, full
  nested procedure, same recipe).
* **Criterion (P):** the pipeline is free of leakage if the mean null AUC is
  within 0.5 +- 0.03 for each fast model and within 0.5 +- 0.07 for the
  network, and the observed AUC exceeds the 99th percentile of the null (fast
  models) or the largest null AUC (network). Permutation p-values
  (1 + number of null AUCs at least the observed) / (1 + permutations) are
  reported.

## D3. Incremental value of movement scores beyond controls

* From the stored out-of-fold probabilities (mean over repetitions, as
  logits): second-level logistic regressions of the label on (a) the control
  scores Scene, Time-of-day, Frame-format and Demographics, and (b) the same
  plus one movement score (test-case network; Gait-RF; Depth-RF), over all
  children and within the cropped videos.
* Reported: likelihood-ratio test of (b) against (a); leave-one-out
  cross-validated AUC of (a) and (b) and their difference with a stratified
  bootstrap interval (2,000 resamples, refitting within each resample);
  Spearman correlation between each movement score and each control score
  over all children and within each group.
* **Criterion (P):** a movement score adds information beyond the recording
  controls if the likelihood-ratio p < 0.05 **and** the bootstrap interval of
  the leave-one-out AUC difference excludes 0.

## D4. All-controls model and time of day

* New nested-CV rows under the frozen partition: *Time-of-day* (logistic
  regression on the depth recording's time of day alone); *All-controls*
  (frame format, age, sex, validity pattern, duration, scene and camera,
  tracking quality (D5), depth-recording properties including time of day).
  The first-phase *Covariates* row is renamed "format, validity pattern and
  demographics" and kept.
* **Criterion (P):** within the cropped videos, a movement model counts as
  evidence beyond the recording conditions only if it exceeds All-controls
  with a paired bootstrap interval above 0 (replaces nothing; the first-phase
  criterion against Covariates is reported as before). The second-phase
  depth-corroboration criterion is additionally evaluated against
  Time-of-day alone.

## D5. Tracking quality (replaces "jitter" as a noise floor)

The 9-frame residual at 60 frames per second keeps movement above about
3 Hz, so it measures high-frequency landmark motion, not estimator noise. It
is kept under that name. New per-recording measures, from the prepared
sequences (shoulder widths), median over the landmarks of each region
(face, arms and hands, legs and feet) and overall:

* *second-difference noise*: root mean square of x(t+1) - 2x(t) + x(t-1)
  over triplets of detected frames, divided by sqrt(6) (white-noise
  estimate);
* *segment-length variability*: for upper arms, forearms, thighs and shanks,
  root mean square difference between the segment length and its 9-frame
  moving average, divided by the median segment length (median over the
  eight segments);
* *visibility*: mean MediaPipe visibility of the landmarks of each region
  (D8 re-extraction).

*Tracking-quality* control: logistic regression on these values under the
frozen partition. Criteria as for the other controls (interval convention).

## D6. Split-half reliability and reliability-bounded agreement

* Each descriptor is computed separately from the first and the second half
  of the detected span (video) or of the frames (depth). Split-half
  reliability: Spearman correlation of the halves across the 95 children,
  Spearman-Brown corrected, r = 2 rho / (1 + rho) (negative values set to 0).
* For the seven descriptors measured in both channels: ceiling
  sqrt(r_video r_depth) and the disattenuated correlation
  rho / sqrt(r_video r_depth) when the ceiling is at least 0.2.
* **Criterion (P):** low video-depth agreement is interpreted as
  disagreement only for descriptors whose ceiling is at least 0.5;
  otherwise the comparison is reported as indeterminate.
* The denominator is reported explicitly: robust video descriptors, how many
  of them have a depth counterpart, and how many of those agree.

## D7. Rotation-robust scale

The arrays are rebuilt from the stored pixel landmarks with two alternative
scales instead of the per-frame shoulder distance: (a) torso length, the
distance between the shoulder and the hip midpoints in the same frame;
(b) the shoulder distance smoothed by a running median over 61 detected
frames (about 1 s). The descriptor analysis within the cropped videos
(adjusted for age and sex) and Gait-LR and Gait-RF under the frozen partition
are repeated. **Criterion (P):** a robust descriptor difference is
*scale-robust* if its adjusted effect keeps its sign with a 95% interval
excluding 0 under both alternative scales.

## D8. Pose quality control

* MediaPipe Pose (same version and settings) is run again on the 95 cohort
  videos, keeping per-landmark visibility; the re-extracted coordinates are
  compared with the stored ones (largest absolute difference).
* Frame-level screen: a detected frame is flagged if the hip midpoint moves
  by more than 0.5 shoulder widths from the previous detected frame, if the
  shoulder distance changes by more than 30%, or if the mean visibility of
  the landmarks is below 0.5. A recording is flagged if more than 5% of its
  detected frames are flagged. Reported by group and frame format.
* Overlay review: 30 recordings drawn at random (rng 20261102), three frames
  each (first, middle and last detected frame), landmarks drawn on the
  frame; inspected for a second person, left-right swaps and gross landmark
  errors. The overlays are not published (they show children).

## D9. What the models use

* *Mean-pose* classifier: logistic regression on the mean of the prepared
  positions over detected frames (66 values) under the frozen partition, all
  children and cropped videos.
* *Frame-order shuffle*: every stored checkpoint of the test-case network
  and of the no-gate ablation scores its test children with the detected
  frames randomly permuted in time (the validity pattern unchanged; five
  permutations, rng 20261103 + k); velocity and acceleration are recomputed
  by the model from the shuffled positions. **Criterion (P):** the model
  depends on temporal order if the AUC falls by more than 0.05 with a paired
  bootstrap interval excluding 0.

## D10. Viewpoint and size proxy

The adjusted descriptor regressions within the cropped videos are repeated
with the child's shoulder width in pixels relative to the frame height as a
further covariate; the robust set is re-evaluated (same rule).

## D11. Ages consistent with the original article

Subset of children whose deposit age lies within the ranges given in the
article describing the collection (autistic 4-12 years, typically developing
6-11 years): AUC of every stored model and control in the subset and in the
subset of cropped videos; the adjusted descriptor analysis within the cropped
videos of the subset (adjusted for sex only, and for age and sex).
Descriptive (no criterion).

## D12. Training within the cropped videos

Nested cross-validation within the 60 children with cropped videos only
(three repetitions, five outer and three inner stratified folds, seed
20261104 + r): Gait-LR, Gait-RF, Gait-LR-adjusted, Mean-pose, Depth-LR,
Depth-RF and the controls Scene, Time-of-day, Demographics, Tracking-quality
and All-controls (without frame format). Compared with the AUCs within the
cropped videos of the models trained on all children. The criteria of C3
are applied to these runs as well.

## D13. Attribution beyond gradient times input

* Real data, every stored checkpoint of the test-case network and a
  randomly initialized network of the same architecture: integrated gradients
  at the descriptor layer (zero baseline, 32 steps), summarized as stream and
  region shares and landmark profiles exactly as gradient times input;
  temporal occlusion: each 15-frame block of the aligned sequence is set to
  undetected in turn and the absolute logit change recorded, giving a share
  per block. The C6 real-data criterion (second phase) is applied to the
  integrated-gradient summaries.
* Planted events: the two-block gate is retrained for raise (A = 1) and flap
  (A = 0.6) with the same folds and seeds as the stored runs (which also
  checks that the stored results are reproduced); integrated gradients and
  temporal occlusion give the share on event frames for trained and random
  networks. The C6 planted-event criterion is applied to both.

## D14. File and container metadata

Descriptive: the header fields of every color video (codec, writing library,
frame size, frame rate, any date field) and the file modification times
stored in the Dryad archive for each child's color video, Kinect trajectory
and skeleton videos and skeleton workbook, by group. File times are not
verified recording dates; they are reported as such.

## D15. Multiplicity

The McNemar comparisons of learned selection against the reference detector
and the motion heuristic are exploratory; Holm-adjusted p-values over all of
them are reported alongside the unadjusted values.

## D16. Verification of the analysis code

Independent re-implementations, run as a script with a report: AUC by direct
pairwise counting against the implementation used; the stratified bootstrap
of one interval re-derived; partition integrity (outer test sets disjoint and
covering all children in every repetition; inner folds inside the outer
training children; no child in two folds); duplicate removal recomputed from
the raw files (MD5); stored arrays recomputed from the stored keypoints; the
label-permutation null (D2) and the exact rerun of one outer fold.
