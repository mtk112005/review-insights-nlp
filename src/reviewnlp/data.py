"""Load, de-duplicate, label and split the Amazon Musical Instruments reviews."""
import gzip
import html
import json
import re

import numpy as np
import pandas as pd

from . import config as C

_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def light_clean(text: str) -> str:
    """Minimal cleaning that keeps what sentiment depends on: negations,
    punctuation (!, ?), casing for transformers. Only HTML/URLs/whitespace go."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    return _WS.sub(" ", text).strip()


def rating_to_label(r: float) -> int:
    return 0 if r <= 2 else (1 if r == 3 else 2)


def read_raw(path=C.RAW_PATH) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            rows.append((o.get("asin"), o.get("reviewerID"), o.get("overall"), o.get("unixReviewTime"),
                         o.get("summary", ""), o.get("reviewText", ""), o.get("verified", False),
                         str(o.get("vote", "0")).replace(",", "")))
    return pd.DataFrame(rows, columns=["asin", "reviewer", "rating", "ts", "summary", "review",
                                       "verified", "votes"])


def build(path=C.RAW_PATH) -> pd.DataFrame:
    df = read_raw(path)
    n_raw = len(df)
    df["review"] = df["review"].map(light_clean)
    df["summary"] = df["summary"].map(light_clean)
    df = df[df["review"].str.len() >= 15]
    # The 5-core file repeats the same review across product variants (same
    # reviewer, same text). Keeping duplicates leaks text between train and test.
    df = df.drop_duplicates(subset=["reviewer", "review"])
    df["date"] = pd.to_datetime(df["ts"], unit="s")
    df["label"] = df["rating"].map(rating_to_label).astype(int)
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)
    df["split"] = np.where(df["date"] >= C.TEST_FROM, "test", "train")
    df = df.drop(columns=["ts"]).reset_index(drop=True)
    print(f"[data] {n_raw:,} raw -> {len(df):,} after cleaning/dedup "
          f"({(df.split == 'test').mean():.1%} in test period >= {C.TEST_FROM})")
    return df


def load() -> pd.DataFrame:
    if not C.REVIEWS_PARQUET.exists():
        C.PROCESSED.mkdir(parents=True, exist_ok=True)
        build().to_parquet(C.REVIEWS_PARQUET, index=False)
    return pd.read_parquet(C.REVIEWS_PARQUET)


def eval_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed random sample of the test period (natural class balance)."""
    test = df[df.split == "test"]
    return test.sample(n=min(C.EVAL_SAMPLE, len(test)), random_state=C.SEED)


if __name__ == "__main__":
    d = load()
    print(d.groupby("split")["label"].value_counts(normalize=True).unstack().round(3))
