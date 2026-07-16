"""
config.py
Central configuration for the AI Image Recognition System.
"""

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
 

class Config:
    # Flask
    SECRET_KEY = os.environ.get("SECRET_KEY", "ai-image-recognition-secret-key-2024")
    DEBUG = os.environ.get("FLASK_DEBUG", "True") == "True"

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'history', 'predictions.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Upload settings
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")
    MODEL_FOLDER = os.path.join(BASE_DIR, "model")
    DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")
    LOG_FOLDER = os.path.join(BASE_DIR, "logs")

    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB max upload size

    # Model
    CUSTOM_MODEL_PATH = os.path.join(MODEL_FOLDER, "image_model.h5")
    CLASS_NAMES_PATH = os.path.join(MODEL_FOLDER, "class_names.pkl")
    IMAGE_SIZE = (224, 224)
    TOP_K = 5


def ensure_directories():
    for folder in [
        Config.UPLOAD_FOLDER,
        Config.REPORTS_FOLDER,
        Config.MODEL_FOLDER,
        Config.LOG_FOLDER,
        os.path.join(BASE_DIR, "history"),
    ]:
        os.makedirs(folder, exist_ok=True)
