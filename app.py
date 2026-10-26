"""
app.py
------
Flask web application for the Fake News Detection System.

Routes:
    GET  /            -> News input form
    POST /analyze      -> Runs the ML model + heuristic analysis, saves
                           to history, shows the interactive dashboard
    GET  /history      -> List of previous analyses
    GET  /history/<id>  -> Re-view a saved analysis's dashboard
    POST /history/clear -> Wipe history table
"""

import os
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash

import database
import utils

app = Flask(__name__)
app.secret_key = "bca-minor-project-fake-news-detector"  # fine for a local student project

MODEL_PATH = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

# Load model + vectorizer once at startup
if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
    raise RuntimeError(
        "Model files not found. Run `python train_model.py` first "
        "(after `python generate_dataset.py` if data/news_dataset.csv is missing)."
    )

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

database.init_db()


def run_full_analysis(news_text: str, source: str):
    """Runs ML prediction + all heuristic features and returns one dict."""
    # --- ML Prediction ---
    X = vectorizer.transform([news_text])
    pred_label = model.predict(X)[0]  # 0 = REAL, 1 = FAKE
    proba = model.predict_proba(X)[0]  # [P(REAL), P(FAKE)]
    prediction = "FAKE" if pred_label == 1 else "REAL"
    confidence = float(proba[1] if pred_label == 1 else proba[0]) * 100

    # --- Suspicious Claim Detection ---
    claims, suspicion_score = utils.detect_suspicious_claims(news_text)

    # --- Source Analysis ---
    source_info = utils.analyze_source(source)
    source_penalty = {"low": 25, "unverified": 10, "unknown": 5, "high": -10}[source_info["credibility"]]

    # --- Credibility Score (0 = not credible, 100 = highly credible) ---
    # Base: model's confidence that it's REAL (invert if FAKE)
    base_real_confidence = float(proba[0]) * 100
    credibility_score = base_real_confidence - (suspicion_score * 0.4) - source_penalty
    credibility_score = max(0, min(100, round(credibility_score, 1)))

    # --- Explainability ---
    explanation = utils.explain_prediction(news_text, vectorizer, model)

    # --- Evidence / Fact-check pointers ---
    keywords = utils.extract_keywords(news_text)
    evidence = utils.suggest_evidence_sources(keywords)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 1),
        "claims": claims,
        "suspicion_score": suspicion_score,
        "source_info": source_info,
        "credibility_score": credibility_score,
        "explanation": explanation,
        "evidence": evidence,
        "keywords": keywords,
    }


@app.route("/")
def index():
    stats = database.get_stats()
    return render_template("index.html", stats=stats)


@app.route("/analyze", methods=["POST"])
def analyze():
    news_text = request.form.get("news_text", "").strip()
    source = request.form.get("source", "").strip()

    if not news_text or len(news_text) < 20:
        flash("Please paste a news article of at least 20 characters.")
        return redirect(url_for("index"))

    result = run_full_analysis(news_text, source)

    analysis_id = database.save_analysis(
        news_text=news_text,
        source=source,
        prediction=result["prediction"],
        confidence=result["confidence"],
        credibility_score=result["credibility_score"],
        suspicious_count=len(result["claims"]),
    )

    return render_template(
        "result.html",
        news_text=news_text,
        source=source,
        result=result,
        analysis_id=analysis_id,
    )


@app.route("/history")
def history():
    records = database.get_history()
    stats = database.get_stats()
    return render_template("history.html", records=records, stats=stats)


@app.route("/history/<int:analysis_id>")
def history_detail(analysis_id):
    record = database.get_analysis_by_id(analysis_id)
    if not record:
        flash("Analysis not found.")
        return redirect(url_for("history"))

    # Re-run heuristic breakdown for display (model prediction/scores are
    # already stored, but claim/source/explanation details are regenerated
    # from the saved text so the dashboard can be fully re-rendered).
    result = run_full_analysis(record["news_text"], record["source"])
    return render_template(
        "result.html",
        news_text=record["news_text"],
        source=record["source"],
        result=result,
        analysis_id=record["id"],
        from_history=True,
    )


@app.route("/history/clear", methods=["POST"])
def history_clear():
    database.clear_history()
    flash("History cleared.")
    return redirect(url_for("history"))


if __name__ == "__main__":
   app.run(debug=True, host="0.0.0.0", port=5000)
