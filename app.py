"""
FastAPI application for NSFW Detection API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from pydantic import BaseModel
from typing import Optional
import os
import numpy as np
from PIL import Image
import io
import open_clip
import requests
import torch

from nsfw_detector import predict_nsfw, load_safety_model

app = FastAPI(
    title="NSFW Detector API",
    description="API for detecting NSFW content in images using CLIP-based model",
    version="1.0.0"
)

# Constants
# LAION safety model variants:
# - "ViT-L/14" expects 768-dim CLIP ViT-L/14 embeddings
# - "ViT-B/32" expects 512-dim CLIP ViT-B/32 embeddings
#
# For best compatibility with the published LAION weights, use OpenAI-pretrained CLIP weights.
DEFAULT_CLIP_MODEL = "ViT-L/14"
DEFAULT_OPENCLIP_MODEL = "ViT-L-14"
OPENCLIP_PRETRAINED_TAGS = {
    "ViT-B-32": "openai",            # valid tag for ViT-B-32
    "ViT-L-14": "openai"             # use OpenAI weights for compatibility with LAION safety model
}
# Adjust how we interpret model outputs; set NSFW_SCORE_SOURCE env to override
# Options: auto (default), col0, col1, invert_single, invertcol0, invertcol1, single
NSFW_SCORE_SOURCE = os.environ.get("NSFW_SCORE_SOURCE", "auto").lower()
# Debug flags
DEBUG_INCLUDE_RAW = os.environ.get("NSFW_INCLUDE_RAW", "").lower() in ("1", "true", "yes")
DEBUG_LOG_RAW = os.environ.get("NSFW_DEBUG_RAW", "").lower() in ("1", "true", "yes")

# Global model storage
_clip_model = None
_clip_preprocess = None
_safety_model_loaded = False


class NSFWResponse(BaseModel):
    nsfw_score: float
    raw_scores: Optional[list] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    safety_model_loaded: bool = False


def extract_nsfw_score(raw_scores):
    """
    Convert model output to a single float score.

    Supports outputs shaped as scalar, (N,), (N,1), or (N,2).
    For binary classifiers that return two columns, we treat the last column as the NSFW probability by default.
    You can override behavior via NSFW_SCORE_SOURCE env (auto|col0|col1|invert_single|invertcol0|invertcol1|single).
    """
    scores = np.array(raw_scores, dtype="float32")
    if scores.size == 0:
        raise HTTPException(status_code=500, detail="NSFW model returned no scores")

    strategy = NSFW_SCORE_SOURCE

    # If model returns two columns (e.g., [safe_prob, nsfw_prob]), take the second column.
    if scores.ndim >= 2 and scores.shape[-1] == 2:
        two_col = scores.reshape(-1, 2)[0]
        if strategy == "col0":
            nsfw_prob = two_col[0]
        elif strategy in ("invertcol0",):
            nsfw_prob = 1.0 - two_col[0]
        elif strategy in ("invertcol1",):
            nsfw_prob = 1.0 - two_col[1]
        else:  # default col1
            nsfw_prob = two_col[1]
    else:
        single = scores.reshape(-1)[0]
        if strategy in ("invert_single", "invert"):
            nsfw_prob = 1.0 - single
        else:
            nsfw_prob = single

    return float(nsfw_prob)


def get_clip_model(model_name: str = DEFAULT_OPENCLIP_MODEL, pretrained: str | None = None):
    """Load CLIP model (lazy loading)"""
    global _clip_model, _clip_preprocess

    if pretrained is None:
        # Allow override via env, otherwise pick a known good tag for the given model
        pretrained = os.environ.get("NSFW_CLIP_PRETRAINED") or OPENCLIP_PRETRAINED_TAGS.get(model_name, "openai")

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
        # Match common CLIP embedding convention and LAION training distribution:
        # L2-normalize embeddings to unit length to avoid score saturation.
        embedding = torch.nn.functional.normalize(embedding, dim=-1)
        embedding = embedding.cpu().numpy().astype("float32")

    return embedding


def warm_models():
    """
    Preload CLIP and safety models so the first request does not pay the download/load cost.
    """
    global _safety_model_loaded
    try:
        get_clip_model()
        load_safety_model(DEFAULT_CLIP_MODEL)
        _safety_model_loaded = True
        print("Model warmup completed")
    except Exception as exc:
        print(f"Model warmup failed: {exc}")
        if DEBUG_LOG_RAW:
            import traceback
            traceback.print_exc()


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
        model_loaded=_clip_model is not None,
        safety_model_loaded=_safety_model_loaded
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
        if DEBUG_LOG_RAW:
            print(f"Raw NSFW scores (single): shape={np.array(nsfw_scores).shape}, values={nsfw_scores}")
        nsfw_score = extract_nsfw_score(nsfw_scores)

        return NSFWResponse(
            nsfw_score=round(nsfw_score, 4),
            raw_scores=np.array(nsfw_scores).tolist() if DEBUG_INCLUDE_RAW else None
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
            if DEBUG_LOG_RAW:
                print(f"Raw NSFW scores (batch): shape={np.array(nsfw_scores).shape}, first={nsfw_scores[0] if len(nsfw_scores)>0 else 'n/a'}")

            for i, (filename, score) in enumerate(zip(filenames, nsfw_scores)):
                nsfw_score = extract_nsfw_score(score)
                is_nsfw = nsfw_score >= threshold
                results.append({
                    "filename": filename,
                    "nsfw_score": round(nsfw_score, 4),
                    "is_nsfw": is_nsfw,
                    "threshold": threshold,
                    "message": "NSFW content detected" if is_nsfw else "Safe content",
                    "raw_scores": np.array(score).tolist() if DEBUG_INCLUDE_RAW else None
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


@app.on_event("startup")
def startup_event():
    """Ensure models are preloaded when the app starts"""
    warm_models()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101)
