"""
FastAPI application for NSFW Detection API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from pydantic import BaseModel
from typing import Optional
import numpy as np
from PIL import Image
import io
import open_clip
import requests
import torch

from nsfw_detector import predict_nsfw

app = FastAPI(
    title="NSFW Detector API",
    description="API for detecting NSFW content in images using CLIP-based model",
    version="1.0.0"
)

# Constants
DEFAULT_CLIP_MODEL = "ViT-L/14"
DEFAULT_OPENCLIP_MODEL = "ViT-L-14"

# Global model storage
_clip_model = None
_clip_preprocess = None


class NSFWResponse(BaseModel):
    nsfw_score: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def extract_nsfw_score(raw_scores):
    """
    Convert model output to a single float score.

    Handles outputs shaped as scalar, (N,), or (N, 1).
    """
    scores = np.array(raw_scores, dtype="float32").reshape(-1)
    if scores.size == 0:
        raise HTTPException(status_code=500, detail="NSFW model returned no scores")
    return float(scores[0])


def get_clip_model(model_name: str = DEFAULT_OPENCLIP_MODEL, pretrained: str = "laion2b_s32b_b82k"):
    """Load CLIP model (lazy loading)"""
    global _clip_model, _clip_preprocess

    if _clip_model is None:
        print(f"Loading CLIP model: {model_name} ({pretrained})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        _clip_model = _clip_model.to(device)
        _clip_model.eval()
        print(f"CLIP model loaded on {device}")

    return _clip_model, _clip_preprocess


def get_image_embedding(image: Image.Image):
    """Extract CLIP embedding from image"""
    model, preprocess = get_clip_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
        embedding = embedding.cpu().numpy().astype("float32")

    return embedding


def fetch_image_from_url(image_url: str) -> bytes:
    """Download image bytes from a URL"""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        return response.content
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not fetch image_url: {exc}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=_clip_model is not None
    )


@app.post("/analyze", response_model=NSFWResponse)
async def analyze_image(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None)
):
    """
    Analyze an image for NSFW content. Accepts either a direct URL or an uploaded file.
    """
    try:
        provided_image_url = image_url or request.query_params.get("image_url")

        if file is None and not provided_image_url:
            raise HTTPException(status_code=400, detail="Provide either an image file or image_url")

        if file is not None and provided_image_url:
            raise HTTPException(status_code=400, detail="Provide only one of: file or image_url")

        # Read image bytes
        if provided_image_url:
            contents = fetch_image_from_url(provided_image_url)
        else:
            contents = await file.read()

        # Validate image
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

        # Get CLIP embedding
        embedding = get_image_embedding(image)

        # Predict NSFW score
        nsfw_scores = predict_nsfw(embedding, DEFAULT_CLIP_MODEL)
        nsfw_score = extract_nsfw_score(nsfw_scores)

        return NSFWResponse(
            nsfw_score=round(nsfw_score, 4)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/batch-analyze")
async def batch_analyze_images(
    files: list[UploadFile] = File(...),
    threshold: float = 0.5
):
    """
    Analyze multiple images for NSFW content

    Args:
        files: List of image files to analyze
        threshold: NSFW threshold (default: 0.5)

    Returns:
        List of NSFW scores and classifications
    """
    try:
        results = []
        embeddings = []
        filenames = []

        # Process all images and get embeddings
        for file in files:
            try:
                contents = await file.read()
                image = Image.open(io.BytesIO(contents))
                embedding = get_image_embedding(image)
                embeddings.append(embedding[0])
                filenames.append(file.filename)
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": f"Failed to process: {str(e)}"
                })

        # Batch predict
        if embeddings:
            embeddings_array = np.array(embeddings).astype("float32")
            nsfw_scores = predict_nsfw(embeddings_array, DEFAULT_CLIP_MODEL)

            for i, (filename, score) in enumerate(zip(filenames, nsfw_scores)):
                nsfw_score = extract_nsfw_score(score)
                is_nsfw = nsfw_score >= threshold
                results.append({
                    "filename": filename,
                    "nsfw_score": round(nsfw_score, 4),
                    "is_nsfw": is_nsfw,
                    "threshold": threshold,
                    "message": "NSFW content detected" if is_nsfw else "Safe content"
                })

        return {"results": results, "total": len(files), "processed": len(embeddings)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "NSFW Detector API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/analyze": "Analyze single image (POST)",
            "/batch-analyze": "Analyze multiple images (POST)",
            "/docs": "API documentation (Swagger UI)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101)
