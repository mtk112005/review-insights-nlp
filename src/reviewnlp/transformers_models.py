"""Transformer approaches, all scored on the same eval sample as the baselines.

1. embeddings_lr  - frozen MiniLM sentence embeddings + logistic regression
2. zero_shot      - off-the-shelf 3-class sentiment model (Twitter RoBERTa),
                    no training on our data
3. distilbert_ft  - DistilBERT fine-tuned on a class-balanced sample
                    (CPU budget: 12k reviews, 1 epoch); prior-corrected variant reported too
"""
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from . import config as C
from .data import eval_sample, load
from .evaluate import record

ZERO_SHOT = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMB_CACHE = C.PROCESSED / "embeddings"


def balanced_sample(df: pd.DataFrame, per_class: int, seed=C.SEED) -> pd.DataFrame:
    parts = [g.sample(min(per_class, len(g)), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed)


def encode(texts, name: str):
    from sentence_transformers import SentenceTransformer

    import hashlib

    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    texts = list(texts)
    # Cache key includes a hash of the exact texts, so a changed sample can never
    # silently reuse embeddings that belong to different rows.
    digest = hashlib.sha1("".join(texts).encode("utf-8")).hexdigest()[:12]
    path = EMB_CACHE / f"{name}_{digest}.npy"
    if path.exists():
        return np.load(path)
    model = SentenceTransformer(C.EMBED_MODEL, device="cpu")
    model.max_seq_length = 256
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    np.save(path, emb)
    return emb


def prior_correct(proba: np.ndarray, train_prior: np.ndarray, target_prior: np.ndarray) -> np.ndarray:
    """Bayes prior-shift correction: the model was trained on balanced classes,
    the real stream is ~80% positive. Re-weight posteriors by p_target/p_train."""
    adj = proba * (target_prior / train_prior)
    return adj / adj.sum(axis=1, keepdims=True)


def embeddings_lr(df, ev):
    train = balanced_sample(df[df.split == "train"], 15000)
    t = time.time()
    X_tr, X_ev = encode(train.review, "train_balanced"), encode(ev.review, "eval")
    clf = LogisticRegression(C=2.0, max_iter=3000).fit(X_tr, train.label)
    prior = df[df.split == "train"].label.value_counts(normalize=True).sort_index().to_numpy()
    raw = clf.predict_proba(X_ev)
    joblib.dump(clf, C.MODELS_DIR / "embeddings_lr.joblib")
    record("minilm_embeddings_lr", ev.label, raw.argmax(1),
           params={"encoder": C.EMBED_MODEL, "train_rows": len(train)})
    proba = prior_correct(raw, np.full(3, 1 / 3), prior)
    return record("minilm_embeddings_lr_prior_corrected", ev.label, proba.argmax(1),
                  params={"encoder": C.EMBED_MODEL, "train_rows": len(train)},
                  extra={"train_seconds": round(time.time() - t)})


def zero_shot(ev):
    from transformers import pipeline

    t = time.time()
    clf = pipeline("sentiment-analysis", model=ZERO_SHOT, device=-1, truncation=True, max_length=256)
    out = clf(list(ev.review), batch_size=32)
    mapping = {"negative": 0, "neutral": 1, "positive": 2}
    pred = [mapping[o["label"].lower()] for o in out]
    return record("roberta_zero_shot", ev.label, pred, params={"model": ZERO_SHOT},
                  extra={"inference_seconds": round(time.time() - t)})


def finetune(df, ev, epochs: int = 1):
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                              TrainingArguments)

    torch.manual_seed(C.SEED)
    train = balanced_sample(df[df.split == "train"], C.FINETUNE_TRAIN_PER_CLASS)
    tok = AutoTokenizer.from_pretrained(C.FINETUNE_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        C.FINETUNE_BASE, num_labels=3, id2label=dict(enumerate(C.LABELS)),
        label2id={l: i for i, l in enumerate(C.LABELS)})

    def tokenize(b):
        return tok(b["text"], truncation=True, max_length=C.MAX_LEN)

    ds_tr = Dataset.from_dict({"text": list(train.review), "label": list(train.label)}).map(tokenize, batched=True)
    ds_ev = Dataset.from_dict({"text": list(ev.review), "label": list(ev.label)}).map(tokenize, batched=True)
    out_dir = C.MODELS_DIR / "distilbert_ft"
    args = TrainingArguments(output_dir=str(out_dir / "checkpoints"), num_train_epochs=epochs,
                             per_device_train_batch_size=16, per_device_eval_batch_size=32,
                             learning_rate=3e-5, weight_decay=0.01,
                             warmup_steps=int(0.06 * epochs * -(-len(train) // 16)),  # 6% warmup
                             logging_steps=25, save_strategy="steps", save_steps=100, save_total_limit=1,
                             report_to=[], seed=C.SEED, dataloader_num_workers=0,
                             use_cpu=True)
    t = time.time()
    trainer = Trainer(model=model, args=args, train_dataset=ds_tr, processing_class=tok)
    ckpts = sorted((out_dir / "checkpoints").glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)  # survive interruptions
    train_s = time.time() - t
    logits = trainer.predict(ds_ev).predictions
    proba = torch.softmax(torch.tensor(logits), dim=1).numpy()
    prior = df[df.split == "train"].label.value_counts(normalize=True).sort_index().to_numpy()

    # Headline = raw argmax: the target metric is macro-F1, which rewards minority-class
    # recall. Prior correction (shift toward the ~85% positive base rate) maximises
    # accuracy instead, so it is reported for comparison only.
    m = record("distilbert_ft", ev.label, proba.argmax(1),
               params={"base": C.FINETUNE_BASE, "train_rows": len(train), "epochs": epochs},
               extra={"train_seconds": round(train_s)})
    record("distilbert_ft_prior_corrected", ev.label,
           prior_correct(proba, np.full(3, 1 / 3), prior).argmax(1),
           params={"base": C.FINETUNE_BASE, "prior_correction": True})
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    np.save(C.MODELS_DIR / "train_prior.npy", prior)
    pd.DataFrame({"review": ev.review.values, "rating": ev.rating.values, "label": ev.label.values,
                  "pred": proba.argmax(1),
                  **{f"p_{l}": proba[:, i] for i, l in enumerate(C.LABELS)}}
                 ).to_parquet(C.REPORTS_DIR / "eval_predictions.parquet", index=False)
    return m


def run(which=("embeddings", "zero_shot", "finetune")):
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    df = load()
    ev = eval_sample(df)
    if "embeddings" in which:
        embeddings_lr(df, ev)
    if "zero_shot" in which:
        zero_shot(ev)
    if "finetune" in which:
        finetune(df, ev)


if __name__ == "__main__":
    import sys
    run(tuple(sys.argv[1:]) or ("embeddings", "zero_shot", "finetune"))
