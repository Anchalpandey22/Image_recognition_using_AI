"""
app.py
Main Flask application for the AI Image Recognition System.
"""

import os
import json
import uuid
from datetime import datetime, date

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_file, abort
)
from werkzeug.utils import secure_filename

from config import Config, ensure_directories
from database import db, init_db
from models import Prediction
from utils import (
    allowed_file, is_valid_image, export_history_to_csv,
    generate_pdf_report, now_date_time, logger
)
from predict import predict_image, warm_up, load_model

ensure_directories()

app = Flask(__name__)
app.config.from_object(Config)

init_db(app)

# Load the model once at startup (avoids slow first request)
with app.app_context():
    warm_up()
    _, ACTIVE_MODE = load_model()
    logger.info(f"Application started. Active model mode: {ACTIVE_MODE}")
    logger.info("Server is running at: http://127.0.0.1:5000")



# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def save_upload(file_storage):
    """Securely save an uploaded file with a unique name. Returns the saved path."""
    original_name = secure_filename(file_storage.filename)
    ext = original_name.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file_storage.save(save_path)
    return unique_name, save_path


def run_prediction_pipeline(filename, filepath):
    """Runs prediction and stores the result in the database. Returns dict."""
    top_label, top_confidence, top5, mode = predict_image(filepath, top_k=Config.TOP_K)
    d, t = now_date_time()

    record = Prediction(
        filename=filename,
        prediction=top_label,
        confidence=top_confidence,
        top5_json=json.dumps(top5),
        model_used=mode,
        date=d,
        time=t,
    )
    db.session.add(record)
    db.session.commit()

    return {
        "id": record.id,
        "filename": filename,
        "prediction": top_label,
        "confidence": top_confidence,
        "top5": top5,
        "model_used": mode,
        "date": d,
        "time": t,
    }


# ---------------------------------------------------------------------------
# Routes - Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", active_mode=ACTIVE_MODE)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/result/<int:record_id>")
def view_result(record_id):
    record = Prediction.query.get_or_404(record_id)
    top5 = json.loads(record.top5_json) if record.top5_json else []
    return render_template("result.html", record=record, top5=top5)


@app.route("/history")
def history():
    search_query = request.args.get("q", "").strip()
    query = Prediction.query.order_by(Prediction.id.desc())
    if search_query:
        query = query.filter(Prediction.prediction.ilike(f"%{search_query}%"))
    records = query.all()
    return render_template("history.html", records=records, search_query=search_query)


@app.route("/dashboard")
def dashboard():
    all_predictions = Prediction.query.all()
    total = len(all_predictions)

    today_str = date.today().strftime("%Y-%m-%d")
    today_count = len([p for p in all_predictions if p.date == today_str])

    class_counts = {}
    for p in all_predictions:
        class_counts[p.prediction] = class_counts.get(p.prediction, 0) + 1

    most_predicted = max(class_counts, key=class_counts.get) if class_counts else "N/A"

    # top classes for charts (limit to top 8 for readability)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    chart_labels = [c[0] for c in sorted_classes]
    chart_values = [c[1] for c in sorted_classes]

    avg_confidence = (
        round(sum(p.confidence for p in all_predictions) / total, 2) if total else 0
    )

    recent = Prediction.query.order_by(Prediction.id.desc()).limit(6).all()

    return render_template(
        "dashboard.html",
        total=total,
        today_count=today_count,
        most_predicted=most_predicted,
        avg_confidence=avg_confidence,
        chart_labels=json.dumps(chart_labels),
        chart_values=json.dumps(chart_values),
        recent=recent,
    )


# ---------------------------------------------------------------------------
# Routes - API / Actions
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("No image file received.", "danger")
        return redirect(url_for("index"))

    file = request.files["image"]

    if file.filename == "":
        flash("No file selected.", "danger")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Allowed: jpg, jpeg, png, webp.", "danger")
        return redirect(url_for("index"))

    try:
        filename, filepath = save_upload(file)
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        flash("Failed to save uploaded file.", "danger")
        return redirect(url_for("index"))

    if not is_valid_image(filepath):
        os.remove(filepath)
        flash("The uploaded file is corrupted or not a valid image.", "danger")
        return redirect(url_for("index"))

    try:
        result = run_prediction_pipeline(filename, filepath)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        flash("Prediction failed. Please try a different image.", "danger")
        return redirect(url_for("index"))

    return render_template("index.html", result=result, active_mode=ACTIVE_MODE)


@app.route("/predict-api", methods=["POST"])
def predict_api():
    """JSON API endpoint, used by drag-and-drop / webcam capture via AJAX."""
    if "image" not in request.files:
        return jsonify({"error": "No image file received."}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Unsupported or missing file."}), 400

    try:
        filename, filepath = save_upload(file)
    except Exception as e:
        logger.error(f"Upload failed (api): {e}")
        return jsonify({"error": "Failed to save file."}), 500

    if not is_valid_image(filepath):
        os.remove(filepath)
        return jsonify({"error": "Corrupted or invalid image."}), 400

    try:
        result = run_prediction_pipeline(filename, filepath)
    except Exception as e:
        logger.error(f"Prediction failed (api): {e}")
        return jsonify({"error": "Prediction failed."}), 500

    result["image_url"] = url_for("static_upload", filename=filename)
    return jsonify(result)


@app.route("/uploads/<filename>")
def static_upload(filename):
    safe_name = secure_filename(filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    if not os.path.exists(path):
        abort(404)
    return send_file(path)


@app.route("/history/delete/<int:record_id>", methods=["POST"])
def delete_record(record_id):
    record = Prediction.query.get_or_404(record_id)
    try:
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], record.filename)
        if os.path.exists(upload_path):
            os.remove(upload_path)
    except Exception as e:
        logger.warning(f"Could not delete file for record {record_id}: {e}")

    db.session.delete(record)
    db.session.commit()
    flash("Record deleted.", "success")
    return redirect(url_for("history"))


@app.route("/history/clear", methods=["POST"])
def clear_history():
    try:
        records = Prediction.query.all()
        for r in records:
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], r.filename)
            if os.path.exists(upload_path):
                os.remove(upload_path)
        Prediction.query.delete()
        db.session.commit()
        flash("All history cleared.", "success")
    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
        flash("Failed to clear history.", "danger")
    return redirect(url_for("history"))


@app.route("/history/download-csv")
def download_csv():
    records = Prediction.query.order_by(Prediction.id.desc()).all()
    if not records:
        flash("No history to export.", "warning")
        return redirect(url_for("history"))

    csv_path = os.path.join(app.config["REPORTS_FOLDER"], "prediction_history.csv")
    export_history_to_csv(records, csv_path)
    return send_file(csv_path, as_attachment=True, download_name="prediction_history.csv")


@app.route("/history/report/<int:record_id>")
def download_report(record_id):
    record = Prediction.query.get_or_404(record_id)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], record.filename)
    report_path = os.path.join(
        app.config["REPORTS_FOLDER"], f"prediction_report_{record.id}.pdf"
    )
    generate_pdf_report(record.to_dict(), image_path, report_path)
    return send_file(report_path, as_attachment=True, download_name=f"prediction_report_{record.id}.pdf")


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return render_template("500.html"), 500


@app.errorhandler(413)
def file_too_large(e):
    flash("File too large. Maximum upload size is 8 MB.", "danger")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=Config.DEBUG)
