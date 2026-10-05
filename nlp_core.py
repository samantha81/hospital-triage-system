# shared NLP library (preprocessing + prediction brain).
# imported by train.py and hospital_app.py

import os
import string
from collections import Counter

import numpy as np
import nltk
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

# ============================================================
# 1. CONFIG
# ============================================================
CONFIDENCE_THRESHOLD = 0.50
OTHER_REJECT_THRESHOLD = 0.50
CORPUS_DIR = "corpus"
MODEL_PATH = "department_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"
EXPLAINER_PATH = "explainer_model.joblib"

# ============================================================
# 2. NLTK + VOCABULARY (built once on import)
# ============================================================
for _pkg in ("punkt", "punkt_tab", "averaged_perceptron_tagger_eng",
             "stopwords", "wordnet", "omw-1.4"):
    nltk.download(_pkg, quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def build_medical_vocab():
    """word -> count of how often it appears in the medical corpus."""
    freq = Counter()
    for fn in os.listdir(CORPUS_DIR):
        if not fn.endswith(".txt"):
            continue
        with open(os.path.join(CORPUS_DIR, fn), encoding="utf-8") as f:
            raw = f.read().lower()
        # Tokenize first so "I'm" -> "i","'m" never collapses into junk like "im".
        for w in word_tokenize(raw):
            if w.isalpha() and len(w) > 1:
                freq[lemmatizer.lemmatize(w)] += 1
    return freq


MEDICAL_FREQ = build_medical_vocab()
MEDICAL_VOCAB = set(MEDICAL_FREQ)
KNOWN_WORDS = set(n.lower() for n in wordnet.all_lemma_names()) | MEDICAL_VOCAB
spell = SpellChecker()

# ============================================================
# 3. TEXT PREPROCESSING (used for both corpus and user input)
# ============================================================
def correct_spelling(word):
    """USER-INPUT typo fix: only unknown tokens; prefer a medical-corpus word."""
    if word in KNOWN_WORDS:
        return word
    candidates = spell.candidates(word)
    if not candidates:
        return word
    medical = [c for c in candidates if c in MEDICAL_VOCAB]
    if medical:
        return max(medical, key=lambda c: MEDICAL_FREQ[c])
    known = [c for c in candidates if c in KNOWN_WORDS]
    if known:
        return max(known, key=lambda c: spell.word_frequency[c])
    return word


def preprocess(text, apply_corrections=False):
    """Core cleaning. apply_corrections=True only for messy USER input."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    if apply_corrections:
        tokens = [correct_spelling(w) for w in tokens]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    tokens = [w for w in tokens if len(w) > 1 and w in KNOWN_WORDS]
    return tokens


def filter_by_pos(processed):
    """Keep only content words (nouns/adjectives/verbs); preserves NEG_ tags."""
    if not processed:
        return []
    original_words, neg_flags = [], []
    for w in processed:
        if w.startswith("NEG_"):
            original_words.append(w[4:]); neg_flags.append(True)
        else:
            original_words.append(w); neg_flags.append(False)
    pos_tags = nltk.pos_tag(original_words)
    relevant_pos = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS",
                    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
    out = []
    for (word, pos), is_neg in zip(pos_tags, neg_flags):
        if pos in relevant_pos:
            out.append(f"NEG_{word}" if is_neg else word)
    return out


def handle_negation(POS_tagged):
    """Prefix words in a negation's scope with NEG_ (e.g. 'no fever' -> 'NEG_fever')."""
    expanded = ["not" if t in ("n't", "nt") else t for t in POS_tagged]
    result, negated = [], False
    negation_words = {"not", "no", "never", "without", "none", "cannot"}
    end_scope = {"and", "or", "but", ".", "!", "?", ",", ";"}
    for token in expanded:
        if token in negation_words:
            negated = True; result.append(token)
        elif negated and token not in end_scope and token.isalpha():
            result.append(f"NEG_{token}"); negated = False
        else:
            if token in end_scope:
                negated = False
            result.append(token)
    return result


def preprocess_single_text(text, apply_corrections=False):
    """Full pipeline -> a cleaned string ready for the TF-IDF vectorizer."""
    tokens = preprocess(text, apply_corrections=apply_corrections)
    return " ".join(handle_negation(filter_by_pos(tokens)))


# ============================================================
# 4. TRAINED MODEL (loaded on first prediction)
# ============================================================
_ARTIFACTS = None


def _artifacts():
    """Load model + vectorizer (+ explainer) once, on first use."""
    global _ARTIFACTS
    if _ARTIFACTS is None:
        vectorizer = joblib.load(VECTORIZER_PATH)
        try:
            explainer = joblib.load(EXPLAINER_PATH)
        except Exception:
            explainer = None
        _ARTIFACTS = {
            "clf": joblib.load(MODEL_PATH),
            "vectorizer": vectorizer,
            "explainer": explainer,
        }
    return _ARTIFACTS


# ============================================================
# 5. PREDICTION HELPERS
# ============================================================
def validate_input(cleaned_text, min_total_freq=5):
    """True if the cleaned text contains enough known corpus words to be a
    real medical complaint (filters out empty/unknown gibberish)."""
    vectorizer = _artifacts()["vectorizer"]
    X_tfidf = vectorizer.transform([cleaned_text])
    feature_indices = X_tfidf.nonzero()[1]
    if len(feature_indices) == 0:
        return False
    n_samples = vectorizer.idf_.shape[0]
    idf = vectorizer.idf_
    total_freq = 0
    for idx in feature_indices:
        df = int((n_samples + 1) / np.exp(idf[idx] - 1) - 1)
        if df > 0:
            total_freq += df
    return total_freq >= min_total_freq


def explain_prediction(cleaned_text, dept, top_k=5):
    """Top words that pushed the input toward `dept` -> [(word, contribution)]."""
    art = _artifacts()
    explainer, vectorizer = art["explainer"], art["vectorizer"]
    if explainer is None or dept not in explainer.classes_:
        return []
    vec = vectorizer.transform([cleaned_text])
    if vec.nnz == 0:
        return []
    class_idx = list(explainer.classes_).index(dept)
    coefs = explainer.coef_[class_idx]
    contribs = []
    feature_names = vectorizer.get_feature_names_out()
    for j in vec.nonzero()[1]:
        contribution = vec[0, j] * coefs[j]
        if contribution > 0:
            contribs.append((feature_names[j], float(contribution)))
    contribs.sort(key=lambda x: x[1], reverse=True)
    return contribs[:top_k]


def predict(text):
    """Full triage prediction for USER input. Returns a dict outcome."""
    art = _artifacts()
    clf, vectorizer = art["clf"], art["vectorizer"]

    cleaned = preprocess_single_text(text, apply_corrections=True)
    if not validate_input(cleaned, min_total_freq=5):
        return {"status": "no_symptoms"}

    probs = clf.predict_proba(vectorizer.transform([cleaned]))[0]
    ranked_full = sorted(zip(clf.classes_, probs), key=lambda x: x[1], reverse=True)

    if ranked_full[0][0] == "other" and ranked_full[0][1] >= OTHER_REJECT_THRESHOLD:
        return {"status": "rejected"}

    ranked = [(d, float(p)) for d, p in ranked_full if d != "other"][:5]
    return {
        "status": "ok",
        "ranked": ranked,
        "low_confidence": ranked[0][1] < CONFIDENCE_THRESHOLD,
        "reasons": explain_prediction(cleaned, ranked[0][0]),
    }
