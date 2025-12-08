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
        model_url = "https://github.com/LAION-AI/CLIP-based-NSFW-Detector/releases/download/v1.0/clip_autokeras_binary_nsfw.zip"
    elif clip_model == "ViT-B/32":
        model_dir = os.path.join(cache_folder, "clip_autokeras_nsfw_b32")
        dim = 512
        model_url = "https://github.com/LAION-AI/CLIP-based-NSFW-Detector/releases/download/v1.0/clip_autokeras_nsfw_b32.zip"
    else:
        raise ValueError(f"Unknown clip model: {clip_model}")

    if not os.path.exists(model_dir):
        print(f"Downloading NSFW model for {clip_model}...")
        zip_path = os.path.join(cache_folder, f"{os.path.basename(model_dir)}.zip")

        urllib.request.urlretrieve(model_url, zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cache_folder)

        os.remove(zip_path)
        print(f"Model downloaded and extracted to {model_dir}")

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
