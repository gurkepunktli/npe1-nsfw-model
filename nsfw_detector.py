"""
CLIP-based NSFW Detector
Based on LAION-AI/CLIP-based-NSFW-Detector
"""

import os
import numpy as np
from functools import lru_cache
from pathlib import Path
import urllib.request
import zipfile

# Reduce TensorFlow logging noise by default; can be overridden via env var
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Allow overriding model download URLs; includes fallbacks (GitHub + HuggingFace)
# Optionally prefer a local mirror first (can include port and/or path prefix).
_MIRROR_BASE_URL = os.environ.get("NSFW_MODEL_MIRROR_BASE_URL")


def _mirror_url(filename: str):
    if not _MIRROR_BASE_URL:
        return None
    return _MIRROR_BASE_URL.rstrip("/") + "/" + filename.lstrip("/")


MODEL_URLS = {
    "ViT-L/14": [
        os.environ.get("NSFW_MODEL_URL_VIT_L14"),
        _mirror_url("clip_autokeras_binary_nsfw.zip"),
        "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_binary_nsfw.zip",
        "https://github.com/LAION-AI/CLIP-based-NSFW-Detector/raw/main/clip_autokeras_binary_nsfw.zip",
        "https://huggingface.co/LAION/CLIP-based-NSFW-Detector/resolve/main/clip_autokeras_binary_nsfw.zip",
    ],
    "ViT-B/32": [
        os.environ.get("NSFW_MODEL_URL_VIT_B32"),
        _mirror_url("clip_autokeras_nsfw_b32.zip"),
        "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_nsfw_b32.zip",
        "https://github.com/LAION-AI/CLIP-based-NSFW-Detector/raw/main/clip_autokeras_nsfw_b32.zip",
        "https://huggingface.co/LAION/CLIP-based-NSFW-Detector/resolve/main/clip_autokeras_nsfw_b32.zip",
    ],
}


def get_cache_folder():
    """Get or create cache folder for models"""
    cache_folder = os.path.join(os.path.dirname(__file__), "models")
    Path(cache_folder).mkdir(parents=True, exist_ok=True)
    return cache_folder


@lru_cache(maxsize=None)
def load_safety_model(clip_model="ViT-L/14"):
    """Load the safety model"""
    import autokeras as ak
    from tensorflow.keras.models import load_model

    cache_folder = get_cache_folder()

    if clip_model == "ViT-L/14":
        model_dir = os.path.join(cache_folder, "clip_autokeras_binary_nsfw")
        dim = 768
        model_urls = [u for u in MODEL_URLS["ViT-L/14"] if u]
    elif clip_model == "ViT-B/32":
        model_dir = os.path.join(cache_folder, "clip_autokeras_nsfw_b32")
        dim = 512
        model_urls = [u for u in MODEL_URLS["ViT-B/32"] if u]
    else:
        raise ValueError(f"Unknown clip model: {clip_model}")

    if not os.path.exists(model_dir):
        if not model_urls:
            raise RuntimeError(
                f"No download URLs configured for {clip_model}. "
                "Set NSFW_MODEL_URL_VIT_L14/NSFW_MODEL_URL_VIT_B32 or pre-populate the models folder."
            )
        print(f"Downloading NSFW model for {clip_model} ...")
        zip_path = os.path.join(cache_folder, f"{os.path.basename(model_dir)}.zip")

        last_error = None
        for url in model_urls:
            print(f"Attempting download: {url}")
            try:
                urllib.request.urlretrieve(url, zip_path)
                break
            except Exception as exc:
                last_error = exc
                print(f"Download failed from {url}: {exc}")
                continue
        else:
            raise RuntimeError(
                f"Failed to download NSFW model ({clip_model}) from configured URLs. "
                "Set NSFW_MODEL_URL_VIT_L14/NSFW_MODEL_URL_VIT_B32 to a reachable mirror or "
                "manually place the model files into the cache folder."
            ) from last_error

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cache_folder)

        os.remove(zip_path)
        print(f"Model downloaded and extracted to {model_dir}")

    # The saved AutoKeras models include optimizer state/config; across TF/Keras versions this can break
    # deserialization (e.g. deprecated optimizer args like `decay`). We only need inference, so skip compile.
    try:
        loaded_model = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS, compile=False)
    except TypeError:
        # Backward compatibility with older TF/Keras that may not support `compile=`.
        loaded_model = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS)
    # Warmup prediction
    loaded_model.predict(np.random.rand(10**2, dim).astype("float32"), batch_size=10**2)

    return loaded_model


def predict_nsfw(embeddings, clip_model="ViT-L/14"):
    """
    Predict NSFW scores for given CLIP embeddings

    Args:
        embeddings: numpy array of CLIP embeddings (N, dim)
        clip_model: CLIP model type ("ViT-L/14" or "ViT-B/32")

    Returns:
        numpy array of NSFW scores between 0 (safe) and 1 (NSFW)
    """
    safety_model = load_safety_model(clip_model)
    nsfw_values = safety_model.predict(embeddings, batch_size=embeddings.shape[0])
    return nsfw_values
