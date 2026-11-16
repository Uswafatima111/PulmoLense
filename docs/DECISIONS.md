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