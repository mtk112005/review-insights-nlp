"""TF-IDF + logistic regression baselines.

* legacy   - the original project's recipe: aggressive cleaning (stopwords,
             including "not"/"no", and punctuation removed) + word 1-2 grams.
* improved - light cleaning that keeps negations and "!", word 1-3 grams,
             sublinear tf, C tuned on a time-based validation year.
"""
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from . import config as C
from .data import eval_sample, load
from .evaluate import record


def legacy_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, max_features=50000)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


def improved_pipeline(c: float = 2.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 3), min_df=3, max_df=0.9,
                                  sublinear_tf=True, max_features=300000,
                                  token_pattern=r"(?u)\b\w[\w']*\b|[!?]")),
        ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=3000)),
    ])


def tune_c(train) -> float:
    fit, val = train[train.date < "2016-01-01"], train[train.date >= "2016-01-01"]
    scores = {}
    for c in (0.5, 1.0, 2.0, 4.0):
        p = improved_pipeline(c).fit(fit.review, fit.label)
        scores[c] = f1_score(val.label, p.predict(val.review), average="macro")
    print("[improved] validation macro-F1 by C:", {k: round(v, 3) for k, v in scores.items()})
    return max(scores, key=scores.get)


def run():
    from .legacy_clean import clean_text

    df = load()
    train, ev = df[df.split == "train"], eval_sample(df)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    t = time.time()
    legacy = legacy_pipeline().fit(train.review.map(clean_text), train.label)
    record("tfidf_legacy", ev.label, legacy.predict(ev.review.map(clean_text)),
           params={"cleaning": "stopwords+punct removed", "ngrams": "1-2"},
           extra={"train_seconds": round(time.time() - t)})

    best_c = tune_c(train)
    t = time.time()
    improved = improved_pipeline(best_c).fit(train.review, train.label)
    record("tfidf_improved", ev.label, improved.predict(ev.review),
           params={"cleaning": "light (negations kept)", "ngrams": "1-3", "C": best_c},
           extra={"train_seconds": round(time.time() - t)})
    joblib.dump(improved, C.MODELS_DIR / "tfidf_improved.joblib")

    # What does dropping stopwords do to negation? Show it on a toy pair.
    for s in ["This is not good at all", "This is good"]:
        print(f"   legacy clean: {s!r} -> {clean_text(s)!r}")
    vec, clf = improved.named_steps["tfidf"], improved.named_steps["clf"]
    vocab = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]  # negative-class weights
    print("[improved] strongest negative n-grams:", ", ".join(vocab[np.argsort(-coefs)[:15]]))
    same_budget()


if __name__ == "__main__":
    run()


def same_budget():
    """Fair comparison with the fine-tuned transformer: TF-IDF trained on the
    exact same class-balanced sample DistilBERT saw (CPU budget), not on all
    142k training reviews."""
    from .transformers_models import balanced_sample

    df = load()
    ev = eval_sample(df)
    train = balanced_sample(df[df.split == "train"], C.FINETUNE_TRAIN_PER_CLASS)
    model = improved_pipeline(2.0).fit(train.review, train.label)
    return record("tfidf_improved_same_6k", ev.label, model.predict(ev.review),
                  params={"train_rows": len(train), "note": "same sample as distilbert_ft"})
