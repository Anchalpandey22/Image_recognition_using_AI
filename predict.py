"""
predict.py
Loads the image recognition model and runs predictions.

Behaviour:
- If a custom-trained model exists at model/image_model.h5 (produced by train_model.py
  on your own dataset/ folders), it is loaded and used.
- Otherwise, the system automatically falls back to a pretrained MobileNetV2 (ImageNet,
  1000 classes) so the application is fully functional immediately after installation,
  with no training required.
"""

import os
import pickle
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input as mobilenet_preprocess,
    decode_predictions as mobilenet_decode,
)
from tensorflow.keras.preprocessing import image as keras_image

from config import Config
from utils import logger

_model = None
_class_names = None
_mode = None  # "custom" or "pretrained"
_pretrained_model = None


def _load_pretrained():
    global _pretrained_model
    if _pretrained_model is None:
        logger.info("Loading pretrained MobileNetV2 (ImageNet) model...")
        _pretrained_model = MobileNetV2(weights="imagenet")
        logger.info("Pretrained model loaded successfully.")
    return _pretrained_model


def load_model():
    """Load the active model. Prefers a custom-trained model if available."""
    global _model, _class_names, _mode

    if _model is not None:
        return _model, _mode

    if os.path.exists(Config.CUSTOM_MODEL_PATH) and os.path.exists(Config.CLASS_NAMES_PATH):
        try:
            logger.info("Custom trained model found. Loading image_model.h5 ...")
            _model = tf.keras.models.load_model(Config.CUSTOM_MODEL_PATH)
            with open(Config.CLASS_NAMES_PATH, "rb") as f:
                _class_names = pickle.load(f)
            _mode = "custom"
            logger.info(f"Custom model loaded with {len(_class_names)} classes.")
            return _model, _mode
        except Exception as e:
            logger.error(f"Failed to load custom model, falling back to pretrained. Error: {e}")

    _model = _load_pretrained()
    _mode = "pretrained"
    return _model, _mode


def preprocess_image_custom(img_path, target_size):
    img = keras_image.load_img(img_path, target_size=target_size)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr


def preprocess_image_pretrained(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = mobilenet_preprocess(arr)
    return arr


def predict_image(img_path, top_k=5):
    """
    Run prediction on an image file.
    Returns: (top_prediction_label, top_confidence, list_of_top_k_dicts, model_mode)
    """
    model, mode = load_model()

    if mode == "custom":
        target_size = Config.IMAGE_SIZE
        arr = preprocess_image_custom(img_path, target_size)
        preds = model.predict(arr, verbose=0)[0]
        top_indices = preds.argsort()[-top_k:][::-1]
        results = [
            {"label": _class_names[i].replace("_", " ").title(), "confidence": round(float(preds[i]) * 100, 2)}
            for i in top_indices
        ]
    else:
        arr = preprocess_image_pretrained(img_path)
        preds = model.predict(arr, verbose=0)
        decoded = mobilenet_decode(preds, top=top_k)[0]
        results = [
            {"label": label.replace("_", " ").title(), "confidence": round(float(score) * 100, 2)}
            for (_, label, score) in decoded
        ]

    top_label = results[0]["label"]
    top_confidence = results[0]["confidence"]
    return top_label, top_confidence, results, mode


def warm_up():
    """Load the model once at app startup to avoid first-request latency."""
    load_model()
