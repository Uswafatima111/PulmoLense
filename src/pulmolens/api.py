
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .inference import predict_image
from .gradcam import generate_gradcam


app = FastAPI(
    title="PulmoLens API",
    description="Research API for 14-class chest X-ray classification.",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "project": "PulmoLens",
        "status": "running",
        "model": "DenseNet121",
        "classes": 14,
        "purpose": "Research demonstration only",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PNG or JPEG image.",
        )

    try:
        contents = await file.read()

        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image must be smaller than 10 MB.",
            )

        image = Image.open(BytesIO(contents)).convert("RGB")

        result = predict_image(image)

        return result

    except HTTPException:
        raise

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again.",
        )

    finally:
        await file.close()

@app.post("/explain")
async def explain(
    disease: str,
    file: UploadFile = File(...)
):
    if disease not in [
        "Atelectasis", "Cardiomegaly", "Effusion",
        "Infiltration", "Mass", "Nodule", "Pneumonia",
        "Pneumothorax", "Consolidation", "Edema",
        "Emphysema", "Fibrosis", "Pleural_Thickening",
        "Hernia"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Invalid disease category."
        )

    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PNG or JPEG image."
        )

    try:
        contents = await file.read()

        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image must be smaller than 10 MB."
            )

        image = Image.open(BytesIO(contents)).convert("RGB")
        result = generate_gradcam(image, disease)

        import base64
        from io import BytesIO as ImageBuffer

        def encode_image(pil_image):
            buffer = ImageBuffer()
            pil_image.save(buffer, format="PNG")
            return base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")

        return {
            "disease": result["disease"],
            "probability": result["probability"],
            "heatmap_base64": encode_image(result["heatmap"]),
            "overlay_base64": encode_image(result["overlay"]),
            "warning": (
                "Research demonstration only. "
                "Not intended for clinical diagnosis."
            )
        }

    except HTTPException:
        raise

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    finally:
        await file.close()
