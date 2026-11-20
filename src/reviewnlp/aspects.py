"""Aspect-based sentiment: which product aspects drive negative reviews?

Review -> sentences -> aspect tags (lexicon) -> sentence sentiment (fine-tuned
DistilBERT). A star rating says *that* a customer is unhappy; aspects say *why*.
"""
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .data import load

ASPECTS = {
    "sound": r"\b(sound|sounds|tone|tones|noise|noisy|hum|buzz\w*|hiss\w*|volume|distortion)\b",
    "build_quality": r"\b(build|built|quality|flimsy|sturdy|solid|cheaply|plastic|construction)\b",
    "durability": r"\b(broke|broken|break|breaks|died|dead|stopped working|lasted|wore|snapped|cracked|fell apart)\b",
    "price_value": r"\b(price|priced|value|worth|money|expensive|cheap|bargain|overpriced|cost)\b",
    "ease_of_use": r"\b(easy|easier|difficult|hard to|setup|set up|install\w*|instructions|intuitive|tune|tuning)\b",
    "fit_size": r"\b(fit|fits|fitting|size|sized|small|large|tight|loose|bulky)\b",
    "shipping_packaging": r"\b(shipping|shipped|arrived|delivery|package|packaging|box|damaged)\b",
    "support_returns": r"\b(customer service|support|return|returned|refund|warranty|seller|replacement)\b",
}
_SENT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if 10 <= len(s.strip()) <= 400]


def tag_aspects(sentence: str) -> list[str]:
    s = sentence.lower()
    return [a for a, pat in ASPECTS.items() if re.search(pat, s)]


def explode_aspects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, text, date, rating in zip(df.index, df.review, df.date, df.rating):
        for sent in split_sentences(text):
            for asp in tag_aspects(sent):
                rows.append((rid, date, rating, asp, sent))
    return pd.DataFrame(rows, columns=["review_id", "date", "rating", "aspect", "sentence"])


class SentenceSentiment:
    """Fine-tuned DistilBERT, applied to sentences (no prior correction: sentence
    sentiment distribution differs from the review-level one)."""

    def __init__(self, path=C.MODELS_DIR / "distilbert_ft"):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=str(path), tokenizer=str(path),
                             device=-1, truncation=True, max_length=96, top_k=None)

    def __call__(self, sentences) -> np.ndarray:
        out = self.pipe(list(sentences), batch_size=64)
        order = {l: i for i, l in enumerate(C.LABELS)}
        proba = np.zeros((len(out), 3))
        for i, scores in enumerate(out):
            for s in scores:
                proba[i, order[s["label"]]] = s["score"]
        return proba


def run(per_year: int = 2000):
    df = load()
    df = df[df.date.dt.year >= 2012]
    sample = (df.groupby(df.date.dt.year, group_keys=False)
                .apply(lambda g: g.sample(min(per_year, len(g)), random_state=C.SEED)))
    asp = explode_aspects(sample)
    print(f"[aspects] {len(sample):,} reviews -> {len(asp):,} aspect mentions")
    proba = SentenceSentiment()(asp.sentence)
    asp["sentiment"] = np.array(C.LABELS)[proba.argmax(1)]
    asp["p_negative"] = proba[:, 0]

    summary = (asp.groupby("aspect")
                  .agg(mentions=("sentence", "size"),
                       negative_rate=("sentiment", lambda s: (s == "negative").mean()),
                       positive_rate=("sentiment", lambda s: (s == "positive").mean()))
                  .sort_values("negative_rate", ascending=False))
    summary["coverage"] = summary.mentions / len(sample)
    # In 1-2 star reviews, which aspects are mentioned negatively most often?
    low = asp[(asp.rating <= 2) & (asp.sentiment == "negative")]
    n_low = int((sample.rating <= 2).sum())
    summary["share_of_1_2_star_reviews"] = low.groupby("aspect").review_id.nunique() / n_low
    summary = summary.fillna(0)

    trend = (asp.assign(year=asp.date.dt.year).groupby(["year", "aspect"])
                .sentiment.apply(lambda s: (s == "negative").mean()).unstack())

    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    s = summary.sort_values("share_of_1_2_star_reviews")
    ax.barh(s.index.str.replace("_", " "), s.share_of_1_2_star_reviews * 100, color="#e76f51")
    ax.set(xlabel="% of 1-2 star reviews with a negative sentence about the aspect",
           title="What drives negative reviews?")
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "aspect_drivers.png", dpi=130); plt.close(fig)

    examples = {a: g.sort_values("p_negative", ascending=False).sentence.head(3).tolist()
                for a, g in asp.groupby("aspect")}
    out = {"reviews_analysed": len(sample), "aspect_mentions": len(asp),
           "summary": summary.round(4).reset_index().to_dict("records"),
           "negative_rate_by_year": trend.round(4).reset_index().to_dict("records"),
           "most_negative_examples": examples}
    (C.REPORTS_DIR / "aspects.json").write_text(json.dumps(out, indent=2, default=str))
    print(summary.round(3).to_string())


if __name__ == "__main__":
    run()
