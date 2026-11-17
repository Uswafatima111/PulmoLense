# RESEARCH NOTES

## Lessons from the COVID CT papers (methodology only)
- One-vs-rest accuracy can inflate per-class numbers (Garg 2022).
- Selecting the best of many models on reported CV results biases the headline.
- Overlapping confidence intervals should not be called "significant".
- Internal accuracy can fail to transfer externally (Hadj Bouzid 2024).
- Merging datasets can create source confounding (Rahat 2026).

## From the X-ray papers (uploaded)
- CheXNet recipe: DenseNet121, ImageNet init, 224x224, Adam lr 1e-3 with plateau decay, batch 16, horizontal flip only.
- Wang 2017 labeler: overall F1 0.90 but Mass 0.52, Nodule 0.75, Pneumonia 0.77 against OpenI human labels.
- Adebayo 2018: Grad-CAM passes randomisation tests; Guided Grad-CAM does not.
- TorchXRayVision lists an expert-labelled NIH subset (4,376 images).

## Verified (Milestone 1, <date>)
- NIH official split: 86,524 train/val + 25,596 test (test = 2,797 patients, per Majkowska 2019).
- Hernia: 227 images in train/val split (HF dataset card).
- No Finding share differs: 48.3% train/val vs 26.6% test (HF dataset card).
- Licence: no restrictions; attribution required (NIH link, CVPR 2017 citation, NIH Clinical Center credit).
- Expert labels (Majkowska 2019): 4,374 images (2,412 val + 1,962 test), four findings;
  plus an 810-image, 14-finding PA-only set read by five radiologists. Access via Google Form.
- Kaggle: 12 h session cap, ~20 GB output limit. Weekly GPU quota is not in Kaggle docs; check own account.

## Still to verify
- Overlap between expert-label validation images and our training set
- Per-label positive counts in the 810-image set
- Exact test counts per label