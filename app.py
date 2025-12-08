"""
FastAPI application for NSFW Detection API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import numpy as np
from PIL import Image
import io
import open_clip
import torch

from nsfw_detector import predict_nsfw

app = FastAPI(
    title="NSFW Detector API",
    description="API for detecting NSFW content in images using CLIP-based model",
    version="1.0.0"
)

# Global model storage
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


class NSFWResponse(BaseModel):
    nsfw_score: float
    is_nsfw: bool
    threshold: float
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def get_clip_model(model_name: str = "ViT-L-14", pretrained: str = "laion2b_s32b_b82k"):
    """Load CLIP model (lazy loading)"""
    global _clip_model, _clip_preprocess, _clip_tokenizer

    if _clip_model is None:
        print(f"Loading CLIP model: {model_name} ({pretrained})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        _clip_model = _clip_model.to(device)
        _clip_model.eval()
        _clip_tokenizer = open_clip.get_tokenizer(model_name)
        print(f"CLIP model loaded on {device}")

    return _clip_model, _clip_preprocess, _clip_tokenizer


def get_image_embedding(image: Image.Image, clip_model_name: str = "ViT-L/14"):
    """Extract CLIP embedding from image"""
    model, preprocess, _ = get_clip_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
        embedding = embedding.cpu().numpy().astype("float32")

    return embedding


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=_clip_model is not None
    )


@app.post("/analyze", response_model=NSFWResponse)
async def analyze_image(
    file: UploadFile = File(...),
    threshold: float = 0.5,
    clip_model: str = "ViT-L/14"
):
    """
    Analyze an image for NSFW content

    Args:
        file: Image file to analyze (JPEG, PNG, etc.)
        threshold: NSFW threshold (default: 0.5)
        clip_model: CLIP model to use ("ViT-L/14" or "ViT-B/32")

    Returns:
        NSFW score and classification
    """
    try:
        # Validate CLIP model
        if clip_model not in ["ViT-L/14", "ViT-B/32"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid clip_model. Must be 'ViT-L/14' or 'ViT-B/32'"
            )

        # Read and validate image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        # Get CLIP embedding
        embedding = get_image_embedding(image, clip_model)

        # Predict NSFW score
        nsfw_scores = predict_nsfw(embedding, clip_model)
        nsfw_score = float(nsfw_scores[0][0])

        # Classify based on threshold
        is_nsfw = nsfw_score >= threshold

        return NSFWResponse(
            nsfw_score=round(nsfw_score, 4),
            is_nsfw=is_nsfw,
            threshold=threshold,
            message="NSFW content detected" if is_nsfw else "Safe content"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/batch-analyze")
async def batch_analyze_images(
    files: list[UploadFile] = File(...),
    threshold: float = 0.5,
    clip_model: str = "ViT-L/14"
):
    """
    Analyze multiple images for NSFW content

    Args:
        files: List of image files to analyze
        threshold: NSFW threshold (default: 0.5)
        clip_model: CLIP model to use

    Returns:
        List of NSFW scores and classifications
    """
    try:
        if clip_model not in ["ViT-L/14", "ViT-B/32"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid clip_model. Must be 'ViT-L/14' or 'ViT-B/32'"
            )

        results = []
        embeddings = []
        filenames = []

        # Process all images and get embeddings
        for file in files:
            try:
                contents = await file.read()
                image = Image.open(io.BytesIO(contents))
                embedding = get_image_embedding(image, clip_model)
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
            nsfw_scores = predict_nsfw(embeddings_array, clip_model)

            for i, (filename, score) in enumerate(zip(filenames, nsfw_scores)):
                nsfw_score = float(score[0])
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
