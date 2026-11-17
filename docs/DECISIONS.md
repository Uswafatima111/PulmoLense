# DECISIONS

| ID | Date | Decision | Reason | Status |
|----|------|----------|--------|--------|
| D-001 | <date> | Primary task is multi-label CXR on NIH ChestX-ray14 | Portfolio value, scale, relevance | Active |
| D-002 | <date> | COVID CT is no longer the primary task | Direction change | Retired |
| D-003 | <date> | Budget is 0 INR; free tools only | Constraint | Active |
| D-004 | <date> | Patient-level splits only | Prevent leakage | Active |
| D-005 | <date> | Thresholds chosen on validation data only | Avoid test tuning | Active |
| D-006 | <date> | Headline results use ImageNet weights, not X-ray-pretrained | XRV "all" weights may have seen NIH test images | Active |
| D-007 | <date> | Never mix CT and X-ray data | Different modalities | Active |
| D-008 | <date> | Product work comes after model work | Avoid polished UI on weak experiments | Active |
| D-009 | <date> | Keep official 25,596 test set untouched; build validation from the 86,524 pool by patient | Official split has no validation set; avoid reuse of test | Active |
| D-010 | <date> | Expert-label evaluation uses a written label mapping and excludes any training overlap | Label sets differ; avoid contamination | Active |
| D-011 | <date> | Provisional research focus: weak-vs-expert evaluation gap (A) with calibration extension (B) | Fits available expert labels and zero-cost compute | Provisional, pending search C and full reads |