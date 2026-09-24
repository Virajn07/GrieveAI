"""GrieveAI Flask entry point."""

from pathlib import Path
import os
import secrets
from flask import Flask
from dotenv import load_dotenv


def create_app(test_config=None):
    load_dotenv()
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY") or secrets.token_hex(32)
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
    from src.configuration import load_threshold_settings
    configured_thresholds = load_threshold_settings()
    db_path = Path(app.instance_path) / "grieveai.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    database_url = os.getenv("DATABASE_URL", "").strip() or f"sqlite:///{db_path}"
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    threshold = os.getenv("CONFIDENCE_THRESHOLD", "").strip()
    app.config["CONFIDENCE_THRESHOLD"] = float(threshold) if threshold else None
    if app.config["CONFIDENCE_THRESHOLD"] is not None and not 0 <= app.config["CONFIDENCE_THRESHOLD"] <= 1:
        raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1")
    dedupe_threshold = os.getenv("DEDUPE_THRESHOLD", "").strip()
    app.config["DEDUPE_THRESHOLD"] = float(dedupe_threshold) if dedupe_threshold else configured_thresholds["duplicate_similarity_threshold"]
    if not 0 <= app.config["DEDUPE_THRESHOLD"] <= 1:
        raise ValueError("DEDUPE_THRESHOLD must be between 0 and 1")
    if test_config:
        app.config.update(test_config)

    from app.models_db import db
    db.init_app(app)
    from app.routes import bp as main_bp
    app.register_blueprint(main_bp)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1")
