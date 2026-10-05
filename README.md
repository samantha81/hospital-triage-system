# Hospital SmartTriage

An NLP-based hospital triage system for Hospital. A patient describes their symptoms in plain
language, and the system classifies the complaint into the correct hospital
department (cardiology, dermatology, neurology, orthopedic, otorhinolaryngology),
rejects non-medical input, and explains its decision.

---

## 1. Project structure

```
project/
├── nlp_core.py            # Shared NLP "brain": text preprocessing + prediction
├── train.py               # Training pipeline: builds & evaluates the model
├── hospital_app.py        # Streamlit web app (patient + staff interface)
│
├── corpus/                # Training data — one .txt file per department
│   ├── cardiology.txt
│   ├── dermatology.txt
│   ├── neurology.txt
│   ├── orthopedic.txt
│   └── otorhinolaryngology.txt
│
├── department_model.joblib   # Saved trained classifier      (created by train.py)
├── vectorizer.joblib         # Saved TF-IDF vectorizer        (created by train.py)
├── explainer_model.joblib    # Saved explainer model          (created by train.py)
│
├── ARCHITECTURE.md        # System architecture diagrams & explanation
└── README.md              # This file
```

---

## 2. What each Python file does

| File | Role | Description |
|------|------|-------------|
| **`nlp_core.py`** | Shared library (the engine) | Defines the text preprocessing pipeline (tokenize, stopword removal, spell-correction, lemmatization, POS filtering, negation handling) and the `predict()` function. Imported by **both** `train.py` and `hospital_app.py` so text is processed identically during training and live use. **You do not run this file directly.** |
| **`train.py`** | Training & evaluation | Loads the corpus, builds a non-medical `"other"` class, extracts TF-IDF features, compares 4 models (Logistic Regression, Linear SVM, Naive Bayes, Random Forest), trains the final model, and saves the 3 `.joblib` files. Also generates evaluation charts (model comparison, metrics summary, confusion matrix, confidence distribution, out-of-scope rejection). **Run once** to produce the model. |
| **`hospital_app.py`** | Web application | The Streamlit interface. Loads the saved `.joblib` model and serves two views: a **Patient** flow (check-in → describe symptoms → referral slip) and a **Staff** dashboard (charts, triage log, manual department reassignment). **This is the main app to run.** |

**Flow:** `train.py` *creates* the model files → `nlp_core.py` *loads & uses* them → `hospital_app.py` *displays* the results.

---

## 3. Requirements

- **Python 3.9+**
- Install the libraries:

```bash
pip install streamlit scikit-learn nltk pandas altair joblib numpy pyspellchecker
```

> The required NLTK data packages (punkt, stopwords, wordnet, etc.) are downloaded
> **automatically** the first time you run the code — no manual NLTK setup needed.
> An internet connection is required on the first run.

---

## 4. How to run

Open a terminal **inside the project folder** (the folder containing these files).

### Step 1 — Train the model (first-time setup)
The web app needs the 3 `.joblib` model files to run. The **first time** you use
the project these don't exist yet, so run the training script once to create them:

```bash
python train.py
```

This loads the corpus, trains the model, and saves `department_model.joblib`,
`vectorizer.joblib`, and `explainer_model.joblib` (plus the evaluation charts as
`.png` images).

You only need to do this **once** — after the files are created, skip straight to
Step 2 every time you use the app. Re-run `train.py` only if you change the corpus
or want to retrain the model.

### Step 2 — Launch the web app

```bash
streamlit run hospital_app.py
```