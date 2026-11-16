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

## To verify (do not cite until checked)
- Official NIH split counts
- Hernia image count
- Licence and patient coverage of the expert-labelled subset