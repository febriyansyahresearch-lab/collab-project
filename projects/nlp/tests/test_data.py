import pytest
from projects.nlp.src.data import generate_texts


def test_generate_texts_shape():
    df = generate_texts(150)
    assert df.shape == (150, 2)


def test_generate_texts_labels():
    df = generate_texts(300)
    assert set(df["label"].unique()) == {"tech", "sports", "finance"}


def test_generate_texts_nonempty():
    df = generate_texts(100)
    assert df["text"].str.len().min() > 0
