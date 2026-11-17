
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = PROJECT_ROOT / "pulmolens_best.pth"

THRESHOLD_PATH = (
    PROJECT_ROOT / "results" / "pulmolens_epoch3_thresholds.npy"
)

DEVICE = torch.device("cpu")


# --------------------------------------------------
# IMAGE PREPROCESSING
# Must match the training preprocessing
# --------------------------------------------------

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

class PulmoLensInference:

    def __init__(self):
        self.device = DEVICE

        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {CHECKPOINT_PATH}"
            )

        if not THRESHOLD_PATH.exists():
            raise FileNotFoundError(
                f"Threshold file not found: {THRESHOLD_PATH}"
            )

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False,
        )

        if checkpoint["diseases"] != DISEASES:
            raise ValueError(
                "Checkpoint disease order does not match configuration."
            )

        self.model = models.densenet121(weights=None)

        self.model.classifier = nn.Linear(
            self.model.classifier.in_features,
            len(DISEASES),
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        self.thresholds = np.load(THRESHOLD_PATH).astype(np.float32)

        if self.thresholds.shape != (len(DISEASES),):
            raise ValueError(
                f"Expected 14 thresholds, got {self.thresholds.shape}"
            )

        self.epoch = checkpoint["epoch"]
        self.validation_auroc = checkpoint["val_macro_auc"]

        print("PulmoLens inference engine initialized.")
        print(f"Model: DenseNet121")
        print(f"Checkpoint epoch: {self.epoch}")
        print(f"Validation macro AUROC: {self.validation_auroc:.4f}")
        print(f"Device: {self.device}")

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    def predict(self, image: Image.Image):

        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL image.")

        image = image.convert("RGB")

        input_tensor = IMAGE_TRANSFORM(image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(input_tensor)
            probabilities = torch.sigmoid(logits)[0].cpu().numpy()

        predictions = []

        for index, disease in enumerate(DISEASES):

            probability = float(probabilities[index])
            threshold = float(self.thresholds[index])

            predictions.append({
                "disease": disease,
                "probability": probability,
                "threshold": threshold,
                "positive": probability >= threshold,
            })

        predictions.sort(
            key=lambda item: item["probability"],
            reverse=True,
        )

        return {
            "predictions": predictions,
            "model": "DenseNet121",
            "checkpoint_epoch": self.epoch,
            "validation_macro_auroc": self.validation_auroc,
            "device": str(self.device),
            "warning": (
                "Research demonstration only. "
                "Not intended for clinical diagnosis."
            ),
        }


# --------------------------------------------------
# SINGLETON MODEL INSTANCE
# --------------------------------------------------

_engine = None


def get_inference_engine():

    global _engine

    if _engine is None:
        _engine = PulmoLensInference()

    return _engine


def predict_image(image: Image.Image):

    engine = get_inference_engine()

    return engine.predict(image)
