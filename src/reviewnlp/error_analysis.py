"""Where does the best model disagree with the stars, and who is wrong?

Star ratings are a noisy label for text sentiment: people write "Great strings,
but shipping took forever" and give 3 stars, or praise a product and give 1 star
by mistake. This module samples the most confident disagreements, and also
measures how a confidence threshold trades coverage for accuracy (useful if
low-confidence reviews go to a human).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C


def run():
    p = pd.read_parquet(C.REPORTS_DIR / "eval_predictions.parquet")
    probs = p[[f"p_{l}" for l in C.LABELS]].to_numpy()
    p["confidence"] = probs.max(1)
    p["correct"] = p.pred == p.label

    # Confusion between extremes: the costly mistakes.
    pos_as_neg = p[(p.label == 2) & (p.pred == 0)].sort_values("p_negative", ascending=False)
    neg_as_pos = p[(p.label == 0) & (p.pred == 2)].sort_values("p_positive", ascending=False)

    # Coverage vs accuracy as we only auto-label confident predictions.
    rows = []
    for t in np.arange(0.4, 0.96, 0.05):
        keep = p.confidence >= t
        rows.append({"threshold": round(float(t), 2), "coverage": float(keep.mean()),
                     "accuracy": float(p.loc[keep, "correct"].mean()) if keep.any() else np.nan})
    curve = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve.coverage * 100, curve.accuracy * 100, marker="o", color="#264653")
    for _, r in curve.iloc[::2].iterrows():
        ax.annotate(f"≥{r.threshold:.2f}", (r.coverage * 100, r.accuracy * 100), fontsize=7,
                    xytext=(4, -10), textcoords="offset points")
    ax.set(xlabel="% of reviews auto-labelled", ylabel="Accuracy on those (%)",
           title="Confidence threshold: coverage vs accuracy")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "coverage_vs_accuracy.png", dpi=130); plt.close(fig)

    out = {
        "eval_reviews": len(p),
        "positive_predicted_negative": int(len(pos_as_neg)),
        "negative_predicted_positive": int(len(neg_as_pos)),
        "coverage_accuracy": curve.round(4).to_dict("records"),
        "examples_5star_text_negative": pos_as_neg.head(5)[["rating", "p_negative", "review"]]
            .assign(review=lambda d: d.review.str[:300]).to_dict("records"),
        "examples_1star_text_positive": neg_as_pos.head(5)[["rating", "p_positive", "review"]]
            .assign(review=lambda d: d.review.str[:300]).to_dict("records"),
    }
    (C.REPORTS_DIR / "error_analysis.json").write_text(json.dumps(out, indent=2, default=float))
    print(curve.round(3).to_string(index=False))
    for k in ("examples_5star_text_negative", "examples_1star_text_positive"):
        print(f"\n--- {k}")
        for e in out[k][:3]:
            print(f"  [{e['rating']:.0f} stars] {e['review'][:200]}")


if __name__ == "__main__":
    run()
