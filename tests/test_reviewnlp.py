import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reviewnlp.aspects import split_sentences, tag_aspects  # noqa: E402
from reviewnlp.data import light_clean, rating_to_label  # noqa: E402
from reviewnlp.topics import npmi_coherence, topic_diversity  # noqa: E402
from reviewnlp.transformers_models import balanced_sample, prior_correct  # noqa: E402


def test_light_clean_keeps_negation_and_punctuation():
    out = light_clean("<b>Not</b> good!!  See http://x.com &amp; more")
    assert "Not" in out and "!!" in out and "http" not in out and "&" in out


def test_rating_to_label():
    assert [rating_to_label(r) for r in [1, 2, 3, 4, 5]] == [0, 0, 1, 2, 2]


def test_aspect_tagging():
    assert "durability" in tag_aspects("The strap broke after a week")
    assert "price_value" in tag_aspects("Great value for the money")
    assert tag_aspects("I like it") == []


def test_sentence_split():
    assert len(split_sentences("Great tone overall. But the jack broke! Would not buy again?")) == 3


def test_prior_correction_moves_toward_target_prior():
    p = np.array([[0.4, 0.3, 0.3]])
    out = prior_correct(p, np.full(3, 1 / 3), np.array([0.1, 0.1, 0.8]))
    assert np.isclose(out.sum(), 1) and out.argmax() == 2


def test_balanced_sample():
    df = pd.DataFrame({"label": [0] * 10 + [1] * 50 + [2] * 500})
    assert balanced_sample(df, 20).label.value_counts().to_dict() == {0: 10, 1: 20, 2: 20}


def test_topic_metrics():
    docs = ["string guitar tune", "string guitar", "cable noise hum", "cable noise"]
    topics = [["string", "guitar"], ["cable", "noise"]]
    assert topic_diversity(topics) == 1.0
    assert npmi_coherence(topics, docs) > 0.5
