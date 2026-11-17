
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

from PIL import Image

from .inference import (
    DISEASES,
    IMAGE_TRANSFORM,
    get_inference_engine,
)


def generate_gradcam(image: Image.Image, disease: str):
    """
    Generate a Grad-CAM heatmap and overlay for one disease.

    Returns:
        dict containing:
            disease
            probability
            heatmap (PIL RGB image)
            overlay (PIL RGB image)
    """

    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL image.")

    if disease not in DISEASES:
        raise ValueError(
            f"Unknown disease. Choose one of: {DISEASES}"
        )

    engine = get_inference_engine()
    model = engine.model

    class_idx = DISEASES.index(disease)

    original = image.convert("RGB")
    original_resized = original.resize((224, 224))

    input_tensor = IMAGE_TRANSFORM(original).unsqueeze(0)
    input_tensor = input_tensor.to(engine.device)

    activations = {}
    gradients = {}

    target_layer = model.features.denseblock4

    # Capture activations and their gradients.
    def forward_hook(module, inputs, output):
        activations["value"] = output

        output.register_hook(
            lambda gradient: gradients.update(
                {"value": gradient}
            )
        )

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)

        # Grad-CAM requires gradients, so do not use inference_mode.
        with torch.enable_grad():
            logits = model(input_tensor)
            score = logits[0, class_idx]
            probability = torch.sigmoid(score).item()

            score.backward()

        activation = activations["value"]
        gradient = gradients["value"]

        # Average gradients across spatial dimensions.
        weights = gradient.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps.
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize heatmap to original model input size.
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].detach().cpu().numpy()

        # Normalize to [0, 1].
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Convert heatmap to RGB.
        colormap = matplotlib.colormaps["jet"]
        colored_cam = colormap(cam)[..., :3]

        heatmap = Image.fromarray(
            (colored_cam * 255).astype(np.uint8)
        )

        # Blend heatmap with original X-ray.
        overlay_array = np.array(original_resized).astype(
            np.float32
        )

        heatmap_array = np.array(heatmap).astype(np.float32)

        overlay_array = (
            0.6 * overlay_array + 0.4 * heatmap_array
        )

        overlay_array = np.clip(
            overlay_array, 0, 255
        ).astype(np.uint8)

        overlay = Image.fromarray(overlay_array)

        return {
            "disease": disease,
            "probability": float(probability),
            "heatmap": heatmap,
            "overlay": overlay,
        }

    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
