# RESEARCH PROPOSAL (DRAFT v0.1, <date>)

Status: provisional. No experiments run yet. Literature read from abstracts/snippets only.

## Context
NIH ChestX-ray14 labels are NLP-mined from reports and known to be noisy
(Majkowska et al., Radiology 2019). Radiologist-adjudicated labels exist for a
subset (4,374 images; four findings), plus an 810-image PA-only, 14-finding set
read by five radiologists. Access is via a Google Form.

## Research questions
RQ1. For labels that can be mapped to the expert set, how much do AUROC and
     AUPRC of a baseline DenseNet121 change when scored against NLP labels vs
     expert labels on the SAME images?
RQ2. Does post-hoc calibration (temperature scaling), fitted on NLP-label
     validation data, remain calibrated (ECE, Brier) when evaluated against
     expert labels?

## Hypotheses (to be tested, not assumed)
H1. Metrics against expert labels differ from metrics against NLP labels, and
    the gap varies by finding.
H2. Calibration fitted on NLP labels degrades against expert labels.
Either hypothesis may be false. Null results will be reported.

## Design
- Official 25,596-image test set untouched until the final evaluation.
- Train/val built from the 86,524 pool by patient (GroupShuffleSplit, fixed seed).
- Exclude from training any image that appears in the expert-label VALIDATION set
  (check overlap first).
- Models: DenseNet121 (ImageNet weights), ResNet50 for comparison.
- Save raw logits for every run; all analysis is post-hoc.
- Thresholds and temperature fitted on validation only.
- Metrics: per-label AUROC, AUPRC, ECE, Brier. Patient-level bootstrap CIs.
- Label mapping from NIH labels to expert labels written down BEFORE evaluation.

## Optional (pending literature search C)
Grad-CAM vs NIH bounding boxes with model-randomisation sanity checks.

## Known limitations
- Expert labels cover few findings; some NIH labels have no counterpart.
- Rare labels will have wide confidence intervals.
- Mapping (e.g.