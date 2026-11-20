"""What do unhappy customers complain about? LDA vs BERTopic on 1-2 star reviews.

LDA follows the original project (bag of words, sklearn). BERTopic clusters
MiniLM sentence embeddings (UMAP + HDBSCAN) and names clusters with c-TF-IDF.
Both are scored with topic diversity (from the original project) and NPMI
coherence computed on the same corpus, so the comparison is like-for-like.
"""
import json
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from . import config as C
from .data import load

N_TOPICS_LDA = 12


def topic_diversity(top_terms_per_topic):
    all_terms = [t for terms in top_terms_per_topic for t in terms]
    return len(set(all_terms)) / len(all_terms)


def npmi_coherence(top_terms_per_topic, docs, topn: int = 10) -> float:
    """Average NPMI of top-word pairs using document co-occurrence."""
    vocab = sorted({t for terms in top_terms_per_topic for t in terms[:topn]})
    vec = CountVectorizer(vocabulary=vocab, binary=True, ngram_range=(1, 2))
    X = vec.fit_transform(docs).tocsc()
    n = X.shape[0]
    idx = {w: i for i, w in enumerate(vocab)}
    df = np.asarray(X.sum(0)).ravel()
    scores = []
    for terms in top_terms_per_topic:
        pair_scores = []
        for a, b in combinations([t for t in terms[:topn] if t in idx], 2):
            ia, ib = idx[a], idx[b]
            co = X[:, ia].multiply(X[:, ib]).sum()
            if co == 0 or df[ia] == 0 or df[ib] == 0:
                pair_scores.append(-1.0)
                continue
            p_ab, p_a, p_b = co / n, df[ia] / n, df[ib] / n
            pair_scores.append(np.log(p_ab / (p_a * p_b)) / -np.log(p_ab))
        if pair_scores:
            scores.append(np.mean(pair_scores))
    return float(np.mean(scores))


def run_lda(docs):
    vec = CountVectorizer(stop_words="english", min_df=5, max_df=0.5, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
    X = vec.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=N_TOPICS_LDA, learning_method="batch",
                                    random_state=C.SEED, n_jobs=-1).fit(X)
    vocab = vec.get_feature_names_out()
    return [[vocab[i] for i in comp.argsort()[::-1][:10]] for comp in lda.components_]


def run_bertopic(docs, embeddings):
    from bertopic import BERTopic
    from bertopic.representation import MaximalMarginalRelevance
    from hdbscan import HDBSCAN
    from umap import UMAP

    model = BERTopic(
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=C.SEED),
        hdbscan_model=HDBSCAN(min_cluster_size=60, min_samples=10, metric="euclidean",
                              cluster_selection_method="eom", prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5,
                                         token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"),
        representation_model=MaximalMarginalRelevance(diversity=0.3),
        nr_topics=15, calculate_probabilities=False, verbose=False,
    )
    topics, _ = model.fit_transform(docs, embeddings)
    return model, np.array(topics)


def plot_topics_over_time(neg: pd.DataFrame, names: dict, path):
    share = (neg[neg.topic >= 0].assign(year=neg.date.dt.year)
             .groupby(["year", "topic"]).size().unstack(fill_value=0))
    share = share[share.sum(1) >= 100]
    share = share.div(share.sum(1), axis=0)
    top = share.mean().sort_values(ascending=False).head(6).index
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for t in top:
        ax.plot(share.index, share[t] * 100, marker="o", label=names[t][:40])
    ax.set(xlabel="Year", ylabel="% of negative reviews", title="Complaint themes over time")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def run():
    from .transformers_models import encode

    df = load()
    neg = df[df.label == 0].reset_index(drop=True)
    docs = neg.review.tolist()
    print(f"[topics] {len(docs):,} negative reviews")

    lda_terms = run_lda(docs)
    emb = encode(docs, "negative_reviews")
    model, topics = run_bertopic(docs, emb)
    info = model.get_topic_info()
    info = info[info.Topic >= 0]
    bt_terms = [[w for w, _ in model.get_topic(t)][:10] for t in info.Topic]

    results = {
        "lda": {"n_topics": N_TOPICS_LDA, "diversity": topic_diversity(lda_terms),
                "npmi": npmi_coherence(lda_terms, docs), "topics": lda_terms},
        "bertopic": {"n_topics": len(bt_terms), "diversity": topic_diversity(bt_terms),
                     "npmi": npmi_coherence(bt_terms, docs),
                     "outlier_share": float((topics == -1).mean())},
    }
    neg["topic"] = topics
    names = {int(t): ", ".join(w for w in terms[:4]) for t, terms in zip(info.Topic, bt_terms)}
    table = []
    for t, size in zip(info.Topic, info.Count):
        sub = neg[neg.topic == t]
        ex = sub.loc[sub.review.str.len().between(80, 300)].head(2).review.tolist()
        table.append({"topic": int(t), "name": names[int(t)], "reviews": int(size),
                      "share_of_negative": float(size / len(neg)), "mean_rating": float(sub.rating.mean()),
                      "examples": ex})
    results["bertopic"]["topics"] = table

    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_topics_over_time(neg, names, C.FIGURES_DIR / "complaint_topics_over_time.png")
    (C.REPORTS_DIR / "topics.json").write_text(json.dumps(results, indent=2))
    neg[["asin", "date", "rating", "topic"]].to_parquet(C.REPORTS_DIR / "negative_review_topics.parquet")
    for k in ("lda", "bertopic"):
        print(f"[topics] {k:9s} diversity {results[k]['diversity']:.2f}  NPMI {results[k]['npmi']:.3f}")
    for row in table:
        print(f"   {row['topic']:>2} {row['share_of_negative']:5.1%}  {row['name']}")


if __name__ == "__main__":
    run()
