# Validation criteria and analysis plan (second revision)

This document fixes, before the analyses listed in Part B are run, what each
check of the validation protocol must show for the corresponding output to be
interpreted, and the exact settings of the new analyses. It is committed to the
repository before any of the Part B analyses is run; the commit time is the
reference for "specified in advance".

Criteria are marked **P** (prospective: specified before the analysis it is
applied to was run) or **R** (retrospective: written after the corresponding
results of the first revision had been seen; they are applied unchanged, but
they are not a pre-specification).

Terms. *AUC*: out-of-fold area under the ROC curve. *Chance* for selection or
attribution measures: the value expected if the same number of valid blocks
(frames) were chosen uniformly at random. *Held-out event sequences*: synthetic
sequences with a planted event whose background child was not in training.

## Part A. Pass/fail criteria

### V1-V2. Does a component or the method add discrimination? (R)

* A component of the method (gate, block selection, attention, validity
  masking, onset alignment, pixel units) **improves** discrimination only if the
  paired bootstrap 95% interval of the AUC difference (method minus ablation)
  excludes 0 **and** the corrected resampled t interval excludes 0.
* The method **outperforms** simpler models only if the same holds against the
  best gait-descriptor classifier and the best comparison network.

### V3. Is discrimination evidence about movement? (R for existing controls, P for new ones)

* (R) Discrimination in a subset counts as evidence about movement only if the
  model's AUC exceeds the AUC of the all-covariates control in the same subset,
  with the paired bootstrap 95% interval of the difference excluding 0.
* (P) Scene/camera control (Part B, B4d) and landmark-jitter control (B4c): if
  either reaches an AUC of at least 0.70 within the cropped videos with the lower
  95% bound above 0.5, discrimination within the cropped videos is **not**
  attributed to movement without further adjustment.
* (P) A gait descriptor is reported as a **robust group difference** only if,
  within the cropped videos, (i) its group effect adjusted for age and sex has a
  Benjamini-Hochberg q below 0.05, and (ii) the adjusted effect has the same sign
  with a 95% interval excluding 0 both when the descriptor is computed from
  low-pass-smoothed trajectories and when it is computed from the first 90
  detected frames only. Only robust differences may be compared with the autism
  gait literature, and then only with the caveats of B4.
* (P) If uncropped and cropped autistic children are separated by the gait
  descriptors with an AUC of at least 0.70, results within the cropped videos are
  described as possibly unrepresentative of the autistic group.

### V4. Can the selection mechanism localize a known event? (P for Part B runs, R for the first-revision runs)

For one event type, amplitude and configuration:

* **Informative amplitude.** Before any network is trained, three reference
  detectors (B2) are applied. An amplitude is *informative* if the best
  reference detector reaches an AUC between 0.70 and 0.95. Results at other
  amplitudes are reported but do not count towards pass or fail.
* **Detection.** Mean AUC over the five folds of at least 0.80 and no fold below
  0.70 (*fold-stable*).
* **Localization.** Highest-scoring valid block (frame) on the event in a share
  of held-out event sequences at least 0.25 above chance, with a one-sided
  sign test against chance (per-sequence hit minus chance) p < 0.01; and a
  positive excess coverage (Wilcoxon p < 0.01).
* **Pass** = detection and localization both met. A configuration that passes
  only in some folds is reported as *fold-dependent* and does not pass.
* **Added value over a heuristic.** Learned selection adds localization beyond a
  heuristic if its top-1 hit share exceeds that of the motion-energy selector
  (B1) and of the best reference detector's localization, by a paired
  McNemar test p < 0.05 over held-out event sequences.
* **Remedy beats frame selection** (the comparison asked for at one shoulder
  width): mean detection AUC of the remedy at least that of frame selection with
  the same 30-frame budget, no fold below 0.70, and top-1 hit share higher than
  frame selection's (McNemar p < 0.05).

### V5. Is selection on real data reproducible and learned? (P for B3, R for the first-revision chance comparison)

* (R) Agreement above random selection: mean Jaccard index of selected blocks
  across models above the hypergeometric chance value, Wilcoxon p < 0.01.
* (P) **Learned, not input-driven**: trained-model agreement exceeds the
  agreement among randomly initialized networks of the same architecture by at
  least 0.05 (mean Jaccard), paired Wilcoxon p < 0.01.
* (P) **More than motion energy**: the selection is described as equivalent to
  a motion-energy heuristic if the mean Jaccard index between trained and
  heuristic selections is at least the mean Jaccard index between trained
  models minus 0.02.
* (R) Selection is *mostly trivial* if more than half of the recordings have no
  more valid blocks than the budget.
* Segments selected on real data are interpreted only if V4 passes at an
  informative amplitude **and** both P criteria of V5 are met.

### V6. Can attributions be interpreted? (R)

* On planted events: attribution share on event frames at least 1.5 times
  chance for trained networks and below 1.2 times chance for random weights.
* On real data: summaries by stream and by body region are interpreted only if
  at least one stream or region share differs between trained and randomly
  initialized networks by more than 0.05 and the landmark profiles of trained
  and random networks correlate with a mean Spearman coefficient below 0.5.

## Part B. New analyses and their settings

All new runs use the partition, folds, seeds and training recipe of the first
revision unless stated. Nothing below is tuned on its own results.

### B1. Gate remedies on the planted hand-raise event

Event and folds exactly as in `scripts/synthetic_events.py` (A = 1.0, 1.5, 2.0,
3.0 shoulder widths), budget M = 2 blocks of 15 frames. Configurations:

* `anneal` (temporary larger budget): during training the budget is
  M_e = max(2, round(20 - 18 (e - 1) / 19)) blocks in epoch e (all 20 blocks in
  epoch 1, 2 blocks from epoch 20 on); evaluation always uses M = 2.
* `gumbel` (stochastic exploration): during training, blocks are selected by the
  top M of s_b + G_b with G_b independent standard Gumbel noise (Gumbel-top-k
  sampling); evaluation is deterministic.
* `perturbed` (differentiable relaxation): during training, the selection is the
  average of the hard top-M indicator over 16 Gaussian perturbations
  s + 0.5 Z of the scores, with the perturbed-optimizer gradient estimator, and
  the tokens are the corresponding expected blocks (and expected positional
  encodings); evaluation is the deterministic hard top-M.
* `motion-M2` (heuristic, not learned): blocks are scored by their mean
  motion energy (sum over landmarks of the frame-to-frame displacement of the
  prepared positions, over pairs of valid frames); the top 2 are selected without
  sigmoid weighting; the rest of the network is trained as usual.

Comparators are the first-revision runs `blocks-M2`, `frames-K30`, `blocks-M8`
and `nogate`.

**Remedy selection rule for B2:** the remedy with the highest mean detection AUC
at A = 1.0; ties (within 0.01) broken by the higher minimum fold AUC, then by
simplicity (anneal, gumbel, perturbed).

### B2. Two further planted event types and reference detectors

* `flap`: vertical oscillation of hands and wrists (landmarks 15-22; elbows at
  half the displacement) with amplitude a shoulder widths and a period of 10
  frames (6 Hz), without a raise, under the same 30-frame flat-topped window.
* `freeze`: arrest of movement. Within the window, every landmark is moved
  towards its position at the event's first frame plus the recording's own
  high-frequency residual (the trajectory minus its 9-frame moving average), by
  the fraction f of the flat-topped window, so that movement stops but tracking
  noise is kept.
* Reference detectors, each giving one score per sequence: D1 hand height (the
  first-revision detector); D2 motion energy (largest absolute deviation of a
  30-frame mean of the motion energy from the recording's median); D3 a 6-Hz
  matched filter (largest magnitude of the 6-Hz Fourier coefficient of the mean
  vertical hand position, detrended, over 30-frame windows). Each detector also
  localizes the event by its best window.
* **Amplitude rule:** before any network is trained, the best reference AUC is
  computed for a in {0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0} (flap) and f in {0.2,
  0.4, 0.6, 0.8, 1.0} (freeze); for each event type the two grid values whose
  best reference AUC is closest to 0.75 and to 0.90 are used (ties: the smaller
  value). If no grid value reaches 0.70, the largest is used and the event type
  is reported as not informative.
* Configurations: `blocks-M2`, `frames-K30`, the remedy chosen by the rule in
  B1, `motion-M2` and `nogate`; five folds grouped by background child.

### B3. Selection nulls on real data

For the stored checkpoints of the method (M = 8), M = 4 and frame selection:

* randomly initialized networks of the same architecture (one per checkpoint,
  seeded), whose selections are compared across the nine networks that would
  score each child, exactly as for trained models;
* the motion-energy heuristic (top M blocks by mean motion energy), compared
  with each trained model's selection (Jaccard index), with hypergeometric
  chance;
* agreement between trained models from different repetitions only, to reduce
  the overlap of training sets.

### B4. Movement evidence

* (a) Combined subset: cropped videos, boys and the shared age range together;
  AUC of every model from the stored out-of-fold predictions (1000 stratified
  bootstrap resamples).
* (b) Gait descriptors within the cropped videos: medians by group, AUC with 95%
  bootstrap interval, Mann-Whitney test, and the group coefficient of an
  ordinary least-squares regression of the standardized descriptor on group, age
  and sex (95% bootstrap interval, 2000 resamples), with Benjamini-Hochberg
  correction over the 26 descriptors.
* (c) Noise floor and clip length: per-recording landmark jitter (root mean
  square of the difference between each landmark trajectory and its 9-frame
  moving average over continuous detected stretches, in shoulder widths; median
  over landmarks, separately for face, arms and hands, and legs and feet);
  jitter by group; a jitter control (logistic regression on the four jitter
  values) under the nested partition; descriptors recomputed from trajectories
  smoothed by a 9-frame moving average, and from the first 90 detected frames
  (children whose detected span is at least 90 frames), each with the adjusted
  analysis of (b).
* (d) Scene/camera control: from every 10th frame of each video, mean luminance,
  luminance standard deviation, mean saturation, sharpness (variance of the
  Laplacian), mean luminance of a border band of 5% of the frame, sharpness in a
  box around the nose (half-width half a shoulder width, a blur index for the
  blurred faces), the video's bit rate, and the child's median shoulder width in
  pixels relative to the frame height; logistic regression under the nested
  partition, over all children and within the cropped videos.
* (e) Age- and sex-adjusted descriptor classifier: the gait descriptors
  residualized on age and sex by linear regression fitted on the training
  children of each fold (confound regression), then the Gait-LR pipeline.
* (f) Representativeness: cropped against uncropped autistic children in age,
  sex, detected frames and each model's out-of-fold probability.

### B5. Depth-camera skeletons (exploratory)

The deposit contains, for every child, a skeleton recorded by the depth camera
(`video/8.xlsx`: 25 joints in camera coordinates, metres, about 30 frames per
second, about 1 s of walking). The same device, distance and processing were
used for both groups, so the frame-format difference of the color videos cannot
occur in these data.

* Descriptors (fixed here): step width (median lateral ankle distance),
  walking speed (slope of the spine-base depth over time) divided by leg length
  (hip-to-ankle distance), arm swing (range of wrist depth relative to the
  shoulder), wrist height relative to the shoulder, knee and elbow ranges of
  motion (3-D angles), trunk sway (standard deviation of the lateral offset of
  the spine-shoulder joint from the spine base), head bob (standard deviation of
  head height relative to the spine shoulder), all lengths divided by the
  shoulder width, and jitter as in B4c.
* Recording-property control: number of frames, mean and initial distance to
  the camera, irregularity of frame intervals, and time of day of the recording.
* Classifiers: logistic regression and random forest on the descriptors, and
  logistic regression on the recording properties, under the nested partition,
  evaluated over all children and within the children with cropped videos.
* **Corroboration (P):** the depth-camera skeletons corroborate a movement
  difference if the descriptor classifier's AUC has a lower 95% bound above 0.5
  within the children with cropped videos and exceeds the recording-property
  control's AUC there.

### B6. Invariance statement

The invariance property is restated for stored arrays of fixed length (prepended
undetected frames must not push valid frames past the end of the array), and it
is checked that storing the first 300 frames before onset alignment removed no
detected frame for any of the 95 recordings.
