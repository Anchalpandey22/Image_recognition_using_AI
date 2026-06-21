"""
models.py
Database models for prediction history.
"""

from datetime import datetime
from database import db


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(120), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    top5_json = db.Column(db.Text, nullable=True)  # JSON string of top-5 results
    model_used = db.Column(db.String(50), default="pretrained")
    date = db.Column(db.String(20), nullable=False)
    time = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 2),
            "model_used": self.model_used,
            "date": self.date,
            "time": self.time,
        }
