
from pathlib import Path
from PIL import Image

from inference import get_inference_engine

# Initialize the trained model
engine = get_inference_engine()

# Temporary test: use one of your saved Grad-CAM X-rays
# Replace this path with the actual image path you choose.
IMAGE_PATH = Path("results/pulmolens_gradcam_examples/gradcam_hernia.png")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

image = Image.open(IMAGE_PATH).convert("RGB")

result = engine.predict(image)

print("\nPULMOLENS PREDICTION RESULTS")
print("=" * 45)

for item in result["predictions"]:
    print(
        f'{item["disease"]:22s} '
        f'Probability: {item["probability"]:.4f} '
        f'Predicted positive: {item["positive"]}'
    )

print("\nModel:", result["model"])
print("Checkpoint epoch:", result["checkpoint_epoch"])
print("Validation AUROC:", result["validation_macro_auroc"])
print("\n", result["warning"])
