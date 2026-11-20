"""Review Insights - Streamlit app.

Analyse a review (sentiment + per-aspect sentence sentiment), compare models,
explore complaint topics and aspect drivers. Run `python run_pipeline.py` first.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reviewnlp import config as C  # noqa: E402
from reviewnlp.aspects import split_sentences, tag_aspects  # noqa: E402

st.set_page_config(page_title="Review Insights", layout="wide")


def read_json(name):
    p = C.REPORTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource
def tfidf_model():
    p = C.MODELS_DIR / "tfidf_improved.joblib"
    return joblib.load(p) if p.exists() else None


@st.cache_resource
def distilbert():
    p = C.MODELS_DIR / "distilbert_ft"
    if not p.exists():
        return None
    from transformers import pipeline
    return pipeline("text-classification", model=str(p), tokenizer=str(p), device=-1,
                    truncation=True, max_length=C.MAX_LEN, top_k=None)


def bert_proba(texts):
    out = distilbert()(list(texts), batch_size=32)
    order = {l: i for i, l in enumerate(C.LABELS)}
    proba = np.zeros((len(out), 3))
    for i, scores in enumerate(out):
        for s in scores:
            proba[i, order[s["label"]]] = s["score"]
    return proba


st.title("Review Insights")
st.caption("Amazon Musical Instruments reviews: sentiment, complaint themes and the product aspects behind them.")
t_analyse, t_models, t_topics, t_aspects = st.tabs(
    ["Analyse a review", "Model comparison", "Complaint topics", "Aspect drivers"])

with t_analyse:
    text = st.text_area("Paste a review", height=140, value=(
        "The tuner looks nice and was easy to set up. Sadly the clip broke after two weeks "
        "and customer service never replied. For the price I expected better."))
    options = [n for n, ok in [("Fine-tuned DistilBERT", distilbert() is not None),
                               ("TF-IDF + logistic regression", tfidf_model() is not None)] if ok]
    if not options:
        st.error("No trained models found. Run `python run_pipeline.py`.")
    elif text.strip():
        choice = st.radio("Model", options, horizontal=True)
        if choice.startswith("Fine"):
            proba = bert_proba([text])[0]
        else:
            proba = tfidf_model().predict_proba([text])[0]
        cols = st.columns(3)
        for c, l, p in zip(cols, C.LABELS, proba):
            c.metric(l.title(), f"{p:.0%}")
        st.subheader("Sentence-level aspects")
        sents = split_sentences(text) or [text]
        if distilbert() is not None:
            sp = bert_proba(sents)
            rows = [{"sentence": s, "aspects": ", ".join(tag_aspects(s)) or "-",
                     "sentiment": C.LABELS[int(p.argmax())], "confidence": f"{p.max():.0%}"}
                    for s, p in zip(sents, sp)]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with t_models:
    comp = read_json("model_comparison.json")
    if comp:
        df = pd.DataFrame(comp).T
        cols = [c for c in ["macro_f1", "f1_negative", "f1_neutral", "f1_positive", "accuracy",
                            "train_seconds", "inference_seconds"] if c in df]
        st.write(f"All models scored on the same {C.EVAL_SAMPLE:,} reviews from the test period "
                 f"(reviews written on/after {C.TEST_FROM}; models trained only on earlier reviews).")
        st.dataframe(df[cols].sort_values("macro_f1", ascending=False).style.format("{:.3f}", subset=cols[:5]),
                     width="stretch")
        imgs = sorted(C.FIGURES_DIR.glob("cm_*.png"))
        for row in range(0, len(imgs), 3):
            for c, img in zip(st.columns(3), imgs[row:row + 3]):
                c.image(str(img))

with t_topics:
    tp = read_json("topics.json")
    if tp:
        a, b = st.columns(2)
        a.metric("BERTopic NPMI coherence", f"{tp['bertopic']['npmi']:.3f}",
                 delta=f"{tp['bertopic']['npmi'] - tp['lda']['npmi']:+.3f} vs LDA")
        b.metric("Topic diversity", f"{tp['bertopic']['diversity']:.2f}",
                 delta=f"{tp['bertopic']['diversity'] - tp['lda']['diversity']:+.2f} vs LDA")
        t = pd.DataFrame(tp["bertopic"]["topics"])
        st.dataframe(t[["name", "reviews", "share_of_negative", "mean_rating"]]
                     .style.format({"share_of_negative": "{:.1%}", "mean_rating": "{:.2f}"}),
                     width="stretch", hide_index=True)
        pick = st.selectbox("Example complaints for topic", t.name)
        for ex in t.set_index("name").loc[pick, "examples"]:
            st.markdown(f"> {ex}")
        img = C.FIGURES_DIR / "complaint_topics_over_time.png"
        if img.exists():
            st.image(str(img))

with t_aspects:
    asp = read_json("aspects.json")
    if asp:
        st.write(f"{asp['reviews_analysed']:,} reviews (2012+) → {asp['aspect_mentions']:,} aspect mentions, "
                 "each sentence classified by the fine-tuned model.")
        img = C.FIGURES_DIR / "aspect_drivers.png"
        if img.exists():
            st.image(str(img))
        s = pd.DataFrame(asp["summary"])
        st.dataframe(s.style.format({c: "{:.1%}" for c in s.columns if c not in ("aspect", "mentions")}),
                     width="stretch", hide_index=True)
        pick = st.selectbox("Most negative sentences for aspect", list(asp["most_negative_examples"]))
        for ex in asp["most_negative_examples"][pick]:
            st.markdown(f"> {ex}")
