
from pathlib import Path
from PIL import Image

from .gradcam import generate_gradcam

# Use an existing Grad-CAM example image.
IMAGE_PATH = Path("results/pulmolens_gradcam_examples/gradcam_hernia.png")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH.resolve()}")

image = Image.open(IMAGE_PATH).convert("RGB")

result = generate_gradcam(image, "Hernia")

print("GRAD-CAM TEST SUCCESSFUL")
print("Disease:", result["disease"])
print("Probability:", round(result["probability"], 4))
print("Heatmap size:", result["heatmap"].size)
print("Overlay size:", result["overlay"].size)

# Save generated outputs.
output_dir = Path("results/gradcam_local_test")
output_dir.mkdir(parents=True, exist_ok=True)

result["heatmap"].save(output_dir / "hernia_heatmap.png")
result["overlay"].save(output_dir / "hernia_overlay.png")

print("Saved heatmap and overlay to:", output_dir.resolve())
