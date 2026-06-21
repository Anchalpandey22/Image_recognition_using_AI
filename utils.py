"""
utils.py
Helper utilities: file validation, CSV export, PDF report generation, logging.
"""

import os
import csv
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

import cv2
from fpdf import FPDF

from config import Config


def setup_logger():
    os.makedirs(Config.LOG_FOLDER, exist_ok=True)
    logger = logging.getLogger("ai_image_recognition")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = RotatingFileHandler(
            os.path.join(Config.LOG_FOLDER, "app.log"), maxBytes=1_000_000, backupCount=3
        )
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger


logger = setup_logger()


def allowed_file(filename):
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in Config.ALLOWED_EXTENSIONS


def is_valid_image(filepath):
    """Validate the file is a real, non-corrupted image.

    Uses two independent checks: Pillow (verify() catches truncated/corrupt
    files) and OpenCV (cv2.imread returns None for files it cannot decode,
    which also rules out non-image binaries that slipped past extension
    checks).
    """
    try:
        from PIL import Image

        with Image.open(filepath) as img:
            img.verify()
    except Exception as e:
        logger.warning(f"Invalid/corrupted image rejected (Pillow): {filepath} ({e})")
        return False

    try:
        cv_img = cv2.imread(filepath)
        if cv_img is None:
            logger.warning(f"Invalid/corrupted image rejected (OpenCV): {filepath}")
            return False
    except Exception as e:
        logger.warning(f"OpenCV failed to read image: {filepath} ({e})")
        return False

    return True


def get_image_dimensions(filepath):
    """Returns (width, height) of an image using OpenCV, or None on failure."""
    img = cv2.imread(filepath)
    if img is None:
        return None
    height, width = img.shape[:2]
    return width, height


def export_history_to_csv(predictions, filepath):
    """Write a list of Prediction objects (or dicts) to a CSV file."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Filename", "Prediction", "Confidence (%)", "Model", "Date", "Time"])
        for p in predictions:
            d = p.to_dict() if hasattr(p, "to_dict") else p
            writer.writerow(
                [d["id"], d["filename"], d["prediction"], d["confidence"], d.get("model_used", ""), d["date"], d["time"]]
            )
    return filepath


def generate_pdf_report(prediction_dict, image_path, output_path):
    """Generate a single-prediction PDF report."""
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(70, 30, 150)
    pdf.cell(0, 15, "AI Image Recognition - Prediction Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_draw_color(70, 30, 150)
    pdf.line(10, 28, 200, 28)
    pdf.ln(10)

    if image_path and os.path.exists(image_path):
        try:
            pdf.image(image_path, x=65, w=80)
            pdf.ln(8)
        except Exception as e:
            logger.warning(f"Could not embed image in PDF: {e}")

    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(0, 0, 0)

    rows = [
        ("Image Name", prediction_dict.get("filename", "N/A")),
        ("Prediction", prediction_dict.get("prediction", "N/A")),
        ("Confidence Score", f"{prediction_dict.get('confidence', 0)}%"),
        ("Model Used", prediction_dict.get("model_used", "N/A")),
        ("Date", prediction_dict.get("date", "N/A")),
        ("Time", prediction_dict.get("time", "N/A")),
    ]

    for label, value in rows:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(50, 10, f"{label}:", border=0)
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, str(value), ln=True)

    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")

    pdf.output(output_path)
    return output_path


def now_date_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
