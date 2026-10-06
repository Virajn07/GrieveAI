"""
GrieveAI Flask app entry point.

Run locally:  python app/app.py
Then visit:   http://127.0.0.1:5000/

Working prototype status: /submit runs a REAL pipeline today (PII
redaction -> language ID -> baseline classifier -> duplicate check ->
confidence gate -> LLM summary/routing if keys are set -> saved to DB).
The classifier is the TF-IDF/LogReg baseline (src/baseline_classifier.py)
until the MuRIL+LoRA checkpoint from Colab training is ready to swap in.
"""

from flask import Flask


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "dev-only-change-before-deployment"
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///grieveai.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    from app.models_db import db
    db.init_app(app)

    from app.routes import bp as main_bp
    app.register_blueprint(main_bp)

    with app.app_context():
        db.create_all()

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
