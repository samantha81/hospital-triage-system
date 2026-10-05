# Only run once to train the model. After that, use predict.py to classify new sentences.

# ==========================================================
# 1. IMPORTS
# ==========================================================
import os

import numpy as np
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# ==========================================================
# 2. CONFIG & CONSTANTS
# ==========================================================
RANDOM_SEED = 42

PALETTE_BG = "#F8F9FA"         
PALETTE_MAIN = "#2563EB"      
PALETTE_SOFT = "#93C5FD"     
PALETTE_WARN = "#EF4444"     
PALETTE_DARK = "#1F2937"      
PALETTE_ACC = "#059669"       

# ==========================================================
# 3. NLTK SETUP
# ==========================================================

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
# Generic English corpora used to build the non-medical "other" class
nltk.download('brown')
nltk.download('reuters')

# ==========================================================
# 4. CORPUS & MODEL PATHS
# ==========================================================

try:
    from google.colab import drive

    print("=" * 60)
    print("GOOGLE COLAB DETECTED - Mounting Google Drive...")
    print("=" * 60)
    drive.mount('/content/drive')

    base_path = "/content/drive/MyDrive/NLP_disease_recognition"
    CORPUS_DIR = os.path.join(base_path, "Corpus")
    MODEL_DIR = base_path

    print("Using Google Drive paths:")
    print(f"  Corpus: {CORPUS_DIR}")
    print(f"  Models: {MODEL_DIR}")

    if not os.path.exists(CORPUS_DIR):
        os.makedirs(CORPUS_DIR, exist_ok=True)

except ImportError:
    CORPUS_DIR = "corpus"
    MODEL_DIR = "."
    print("Running in local mode (not Google Colab)")

MODEL_PATH = os.path.join(MODEL_DIR, "department_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
EXPLAINER_PATH = os.path.join(MODEL_DIR, "explainer_model.joblib")

# ==========================================================
# 5. SHARED PREPROCESSING (imported from nlp_core)
# ==========================================================
import nlp_core as core
preprocess_single_text = core.preprocess_single_text
CONFIDENCE_THRESHOLD = core.CONFIDENCE_THRESHOLD   # single source of truth

# ==========================================================
# 6. DATA LOADING (CORPUS + "OTHER" CLASS)
# ==========================================================

def preprocessing(corpus_folder=None):
    """Load each department's .txt file, clean every sentence, return (texts, labels)."""
    corpus_folder = corpus_folder or CORPUS_DIR

    txt_files = sorted([f for f in os.listdir(corpus_folder) if f.endswith('.txt')])

    print(f"Found {len(txt_files)} department files: {', '.join(txt_files)}")

    X_clean, y_labels = [], []
    for filename in txt_files:
        label = os.path.splitext(filename)[0].lower()
        file_path = os.path.join(corpus_folder, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for sentence in nltk.sent_tokenize(f.read()):
                    cleaned = preprocess_single_text(sentence)
                    if cleaned and len(cleaned.split()) >= 2:
                        X_clean.append(cleaned)
                        y_labels.append(label)
        except Exception as e:
            print(f"Warning: Could not read {filename}: {e}")

    return X_clean, y_labels

X_clean, y_labels = preprocessing()
print(f"Loaded {len(X_clean)} sentences")

# --- Build the "other" (non-medical) class ---
# Teaches the model what ordinary non-medical English looks like
# (NLTK brown/reuters), labelled "other", so junk routes there.

def load_other_sentences(n_samples):
    """Sample generic non-medical sentences from NLTK corpora as the 'other' class."""
    raw = []
    for corpus_name in ("brown", "reuters"):
        try:
            corpus = getattr(__import__("nltk.corpus", fromlist=[corpus_name]), corpus_name)
            raw.extend(" ".join(s) for s in corpus.sents())
        except Exception as e:
            print(f"  (skipped {corpus_name}: {e})")

    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(raw)

    out = []
    for sentence in raw:
        cleaned = preprocess_single_text(sentence)
        if cleaned and len(cleaned.split()) >= 2:
            out.append(cleaned)
        if len(out) >= n_samples:
            break
    return out

# Balance: make "other" roughly the size of one department.
n_per_dept = len(X_clean) // max(1, len(set(y_labels)))
print(f"Building 'other' class with ~{n_per_dept} non-medical sentences...")
other_texts = load_other_sentences(n_per_dept)
X_clean.extend(other_texts)
y_labels.extend(["other"] * len(other_texts))
print(f"Added {len(other_texts)} 'other' sentences. Total: {len(X_clean)} sentences")

# ==========================================================
# 7. FEATURE EXTRACTION (TF-IDF)
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(X_clean, y_labels, test_size=0.2, random_state=RANDOM_SEED, stratify=y_labels)

vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"Train shape: {X_train_tfidf.shape}, Test shape: {X_test_tfidf.shape}")

# ==========================================================
# 8. MODEL SELECTION
# ==========================================================
# Compare candidates on the same TF-IDF features, then proceed with the best.

def compare_models(X_tr, y_tr, X_te, y_te, save_path="model_comparison.png"):
    models = {
        "Logistic Regression": LogisticRegression(
            C=2.0, max_iter=1000, random_state=RANDOM_SEED,
            class_weight='balanced', solver='lbfgs'),
        "Linear SVM": LinearSVC(
            C=1.0, class_weight='balanced', random_state=RANDOM_SEED),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED,
            class_weight='balanced', n_jobs=-1),
    }

    print("\n" + "=" * 60)
    print("MODEL SELECTION - comparing candidates (same TF-IDF features)")
    print("=" * 60)
    names, accs, f1s = [], [], []
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, average='macro', zero_division=0)
        names.append(name); accs.append(acc); f1s.append(f1)
        print(f"  {name:22} accuracy={acc:.4f}  macro-F1={f1:.4f}")

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE_BG)
    ax.set_facecolor(PALETTE_BG)
    b1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color=PALETTE_MAIN, edgecolor='white')
    b2 = ax.bar(x + width / 2, f1s, width, label="Macro-F1", color=PALETTE_SOFT, edgecolor='white')
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha='center', fontsize=8, color=PALETTE_DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", color=PALETTE_DARK)
    ax.set_title("Model Comparison (same TF-IDF features)", color=PALETTE_DARK, fontsize=12)
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=PALETTE_DARK)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved model comparison to {save_path}")
    plt.close()

    best = names[int(np.argmax(accs))]
    return best

best_model = compare_models(X_train_tfidf, y_train, X_test_tfidf, y_test)
print(f"\nBest model by accuracy: {best_model}")
print("Proceeding with Logistic Regression (best score + calibrated probabilities + explainable).")

# ==========================================================
# 9. TRAIN FINAL MODEL
# ==========================================================

print("\n" + "=" * 60)
print("TRAINING FINAL MODEL (Logistic Regression + calibration)")
print("=" * 60)

base_clf = LogisticRegression(
    C=2.0, max_iter=1000, random_state=RANDOM_SEED,
    class_weight='balanced', solver='lbfgs'
)
clf = CalibratedClassifierCV(base_clf, cv=5)
clf.fit(X_train_tfidf, y_train)

# Uncalibrated copy, used ONLY to explain predictions: CalibratedClassifierCV hides the word weights, but a plain LogisticRegression exposes coef_, which tells us how much each word pushed toward a department.
explainer_clf = LogisticRegression(
    C=2.0, max_iter=1000, random_state=RANDOM_SEED,
    class_weight='balanced', solver='lbfgs'
)
explainer_clf.fit(X_train_tfidf, y_train)
joblib.dump(explainer_clf, EXPLAINER_PATH)   # used by the web app to explain predictions

joblib.dump(clf, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
print(f"\nModel saved to {MODEL_PATH}")

y_pred = clf.predict(X_test_tfidf)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ==========================================================
# 10. EVALUATION & ANALYSIS
# ==========================================================

# --- 10a. Per-department metrics summary ---
print("Generating metrics summary...")

labels = list(clf.classes_)
overall_acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

dept_names = [d for d in labels if d in report]
precisions = [report[d]['precision'] for d in dept_names]
recalls = [report[d]['recall'] for d in dept_names]
f1s = [report[d]['f1-score'] for d in dept_names]

x, width = np.arange(len(dept_names)), 0.26

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                gridspec_kw={'width_ratios': [2.2, 1]})
fig.patch.set_facecolor(PALETTE_BG)

# Left: per-class bars
ax1.set_facecolor(PALETTE_BG)
ax1.bar(x - width, precisions, width, label='Precision', color=PALETTE_MAIN, edgecolor='white')
ax1.bar(x, recalls, width, label='Recall', color=PALETTE_SOFT, edgecolor='white')
ax1.bar(x + width, f1s, width, label='F1-Score', color=PALETTE_WARN, edgecolor='white')
ax1.set_xticks(x)
ax1.set_xticklabels(dept_names, rotation=35, ha='right', fontsize=9)
ax1.set_ylim(0, 1.15)
ax1.set_ylabel("Score", color=PALETTE_DARK)
ax1.set_title("Per-Department: Precision / Recall / F1", color=PALETTE_DARK, fontsize=12)
ax1.legend(fontsize=9)
ax1.spines[['top', 'right']].set_visible(False)
ax1.tick_params(colors=PALETTE_DARK)

# Right: macro/weighted/accuracy table
ax2.axis('off')
ax2.set_facecolor(PALETTE_BG)
summary_keys = ['macro avg', 'weighted avg']
row_data = [[k.title(), f"{report[k]['precision']:.3f}",
              f"{report[k]['recall']:.3f}", f"{report[k]['f1-score']:.3f}"]
             for k in summary_keys if k in report]
row_data.append(['Accuracy', f"{overall_acc:.3f}", '', ''])
table = ax2.table(cellText=row_data, colLabels=['', 'Prec.', 'Rec.', 'F1'],
                  cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.8)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor(PALETTE_SOFT)
    if r == 0:
        cell.set_facecolor(PALETTE_DARK)
        cell.set_text_props(color='white', fontweight='bold')
ax2.set_title("Summary", color=PALETTE_DARK, fontsize=11)

plt.tight_layout()
fig.savefig("metrics_summary.png", dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved metrics summary to metrics_summary.png")
plt.close()

# --- 10b. Confidence distribution (correct vs incorrect) ---
print("Analyzing confidence distribution...")

y_score = clf.predict_proba(X_test_tfidf)
top1_conf = np.max(y_score, axis=1)

y_test_arr, y_pred_arr = np.array(y_test), np.array(y_pred)
correct_mask = (y_pred_arr == y_test_arr)
correct_conf, wrong_conf = top1_conf[correct_mask], top1_conf[~correct_mask]

fig, ax = plt.subplots(figsize=(7, 4.5))
fig.patch.set_facecolor(PALETTE_BG)
ax.set_facecolor(PALETTE_BG)
bins = np.linspace(0, 1, 21)
ax.hist(correct_conf, bins=bins, color=PALETTE_MAIN, alpha=0.65,
        label=f"Correct (n={len(correct_conf)})", edgecolor='white')
ax.hist(wrong_conf, bins=bins, color=PALETTE_WARN, alpha=0.65,
        label=f"Incorrect (n={len(wrong_conf)})", edgecolor='white')
ax.axvline(CONFIDENCE_THRESHOLD, color=PALETTE_DARK, linestyle='--',
           linewidth=1.5, label=f"Threshold ({CONFIDENCE_THRESHOLD:.0%})")
ax.set_xlabel("Top-1 Predicted Confidence", color=PALETTE_DARK)
ax.set_ylabel("Number of Test Samples", color=PALETTE_DARK)
ax.set_title("Confidence: Correct vs Incorrect Predictions", color=PALETTE_DARK, fontsize=11)
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(colors=PALETTE_DARK)
plt.tight_layout()
fig.savefig("confidence_distribution.png", dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved confidence distribution to confidence_distribution.png")
plt.close()

# Print confidence statistics
print(f"\n  Mean confidence (correct)   : {correct_conf.mean():.4f}")
print(f"  Mean confidence (incorrect) : {wrong_conf.mean():.4f}")
below_thresh = (top1_conf < CONFIDENCE_THRESHOLD).sum()
print(f"  Below threshold ({CONFIDENCE_THRESHOLD:.0%}): {below_thresh}/{len(top1_conf)} ({below_thresh/len(top1_conf):.1%})")

# --- 10c. Confusion matrix ---
def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix", save_path=None):
    """Confusion matrix with counts + per-row percentages."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    n = len(labels)

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    fig.patch.set_facecolor(PALETTE_BG)
    ax.set_facecolor(PALETTE_BG)

    sns.heatmap(cm_pct, annot=False, fmt='', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, cbar=True, linewidths=0.5, linecolor='white')

    for i in range(n):
        for j in range(n):
            colour = 'white' if cm_pct[i, j] > 0.6 else PALETTE_DARK
            ax.text(j + 0.5, i + 0.5,
                    f"{cm[i,j]}\n({cm_pct[i,j]:.0%})",
                    ha='center', va='center', fontsize=8, color=colour)

    ax.set_xlabel("Predicted", color=PALETTE_DARK, labelpad=8)
    ax.set_ylabel("Actual", color=PALETTE_DARK, labelpad=8)
    ax.set_title(title, color=PALETTE_DARK, pad=12)
    plt.xticks(rotation=40, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    plt.close()

print("\nGenerating confusion matrix...")
# Put "other" last so the 5 departments stay grouped together.
cm_labels = [c for c in clf.classes_ if c != "other"] + [c for c in clf.classes_ if c == "other"]
plot_confusion_matrix(y_test, y_pred, cm_labels, title="Disease Recognition - Confusion Matrix", save_path="confusion_matrix.png")

# --- 10d. Out-of-scope (junk) rejection analysis ---
def plot_rejection_analysis(y_true, y_pred, save_path="rejection_analysis.png"):
    """Bar chart of out-of-scope handling: junk rejected vs real symptoms kept."""
    y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)
    true_other, pred_other = (y_true_arr == "other"), (y_pred_arr == "other")

    n_junk, n_med = int(true_other.sum()), int((~true_other).sum())
    junk_rejected = int((true_other & pred_other).sum())     # correctly rejected
    junk_leaked   = int((true_other & ~pred_other).sum())    # missed junk
    med_accepted  = int((~true_other & ~pred_other).sum())   # correctly accepted
    med_rejected  = int((~true_other & pred_other).sum())    # false rejection

    rates = [
        junk_rejected / n_junk if n_junk else 0,
        junk_leaked   / n_junk if n_junk else 0,
        med_accepted  / n_med if n_med else 0,
        med_rejected  / n_med if n_med else 0,
    ]

    print("\n" + "=" * 60)
    print("OUT-OF-SCOPE (JUNK) REJECTION ANALYSIS")
    print("=" * 60)
    print(f"Non-medical correctly rejected : {junk_rejected}/{n_junk} ({rates[0]:.1%})")
    print(f"Non-medical leaked through      : {junk_leaked}/{n_junk} ({rates[1]:.1%})")
    print(f"Real symptoms correctly kept    : {med_accepted}/{n_med} ({rates[2]:.1%})")
    print(f"Real symptoms wrongly rejected  : {med_rejected}/{n_med} ({rates[3]:.1%})")

    labels = ["Junk\nrejected", "Junk\nleaked", "Symptoms\nkept", "Symptoms\nwrongly\nrejected"]
    colors = [PALETTE_ACC, PALETTE_WARN, PALETTE_MAIN, PALETTE_WARN]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(PALETTE_BG)
    ax.set_facecolor(PALETTE_BG)
    bars = ax.bar(labels, rates, color=colors, edgecolor='white')
    for bar, v in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.0%}",
                ha='center', fontsize=9, color=PALETTE_DARK)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Rate", color=PALETTE_DARK)
    ax.set_title("Out-of-Scope Detection (Medical vs Non-medical)",
                 color=PALETTE_DARK, fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=PALETTE_DARK, labelsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved rejection analysis to {save_path}")
    plt.close()

print("\nGenerating out-of-scope rejection analysis...")
plot_rejection_analysis(y_test, y_pred)
