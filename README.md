# Fake News Detection System

**BCA 5th Semester Minor Project** — Machine Learning + Web Development

A Flask web app that analyzes a pasted news article/message and predicts
whether it is **FAKE** or **REAL**, backed by a TF-IDF + Logistic Regression
model, plus explainable heuristic analysis (suspicious claims, source
credibility, evidence pointers) and a SQLite-backed history dashboard.

---

## 1. Features

| Feature | How it works |
|---|---|
| News Input | Paste article text (+ optional source URL) on the home page |
| FAKE/REAL Prediction | TF-IDF vectorizer + Logistic Regression classifier |
| Credibility Score | 0–100 score combining model confidence, suspicious-pattern count, and source trust |
| Suspicious Claim Detection | Rule-based scan for sensational language, clickbait, vague attribution, excessive punctuation/caps, unattributed stats |
| Source Analysis | Domain checked against curated credible / low-credibility pattern lists |
| Evidence Collection | Curated links to real fact-checking sites (PIB Fact Check, Alt News, Boom Live, Snopes, PolitiFact, Google Fact Check Explorer) with an auto-built search query |
| Explainable Result | Shows which specific words in *this* article pushed the verdict toward FAKE or REAL, using the model's own coefficients |
| Analysis History | Every analysis is saved to SQLite (`history.db`) and browsable/re-viewable |
| Interactive Dashboard | All of the above combined on one result page |

## 2. Tech Stack

- **Backend:** Python, Flask
- **ML:** scikit-learn (TF-IDF + Logistic Regression), pandas
- **Database:** SQLite (via built-in `sqlite3`)
- **Frontend:** HTML, CSS (Jinja2 templates, no JS framework needed)

## 3. Project Structure

```
fake_news_detector/
├── app.py                 # Flask app & routes
├── train_model.py         # Trains and saves the ML model
├── generate_dataset.py    # Builds the sample training dataset
├── database.py            # SQLite helper functions
├── utils.py                # Suspicious-claim / source / explainability logic
├── requirements.txt
├── data/
│   └── news_dataset.csv   # Training data (generated)
├── model/
│   ├── model.pkl           # Trained Logistic Regression model
│   └── vectorizer.pkl      # Fitted TF-IDF vectorizer
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   └── history.html
└── static/
    └── css/style.css
```

## 4. Setup & Run

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Already done, but if you want to regenerate)
python generate_dataset.py
python train_model.py

# 4. Run the app
python app.py
```

Then open **http://127.0.0.1:5000/** in your browser.

The `model/model.pkl`, `model/vectorizer.pkl`, and `data/news_dataset.csv`
files are already included and pre-trained, so `python app.py` works
immediately without re-running the training steps.

## 5. About the Training Data

This project ships with a **synthetically generated** dataset
(`data/news_dataset.csv`, 700 rows) built from realistic templates of how
factual/attributed news is written versus sensational/unverified news. This
keeps the whole project runnable **offline**, without needing large external
downloads, which is ideal for a minor project demo/viva.

**To improve real-world accuracy**, replace `data/news_dataset.csv` with a
larger public dataset — for example, the Kaggle "Fake and Real News Dataset"
(`Fake.csv` + `True.csv`). Just make sure the final CSV has two columns:
`text` and `label` (1 = FAKE, 0 = REAL), then re-run:

```bash
python train_model.py
```

No other code changes are needed — `app.py` and `utils.py` work with any
dataset shaped this way.

## 6. Why Logistic Regression + TF-IDF?

- **TF-IDF** turns article text into numeric features weighted by how
  distinctive a word is to a document, which works well for text
  classification without needing deep learning.
- **Logistic Regression** is fast to train, easy to explain (its
  coefficients directly show which words push a prediction toward FAKE or
  REAL), and performs competitively on this kind of task — all of which
  matters for a college-level explainable-AI demo.

## 7. Limitations (good to mention in your viva)

- The bundled dataset is synthetic/template-based, so real accuracy on
  novel, real-world news will be lower than the demo's test accuracy.
- Suspicious-claim and source-credibility checks are **rule-based
  heuristics**, not machine-learned — this is intentional, to keep results
  transparent and explainable.
- "Evidence Collection" surfaces links to real fact-checking sites with a
  suggested search query rather than performing a live automated fact-check
  (no external API key is used, keeping the project fully offline-runnable).
