from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def make_tfidf(kind: str, max_features: int = 50000, min_df: int = 2):
    """Build one of the deliberately small set of text feature configurations."""
    word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        max_features=max_features,
        min_df=min_df,
        ngram_range=(1, 2),
    )
    if kind == "word_unigram":
        word.set_params(ngram_range=(1, 1))
        return word
    if kind == "word_bigram":
        return word
    if kind == "word_char":
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=min_df,
            max_features=max_features // 2,
            sublinear_tf=True,
        )
        word.set_params(max_features=max_features // 2)
        return FeatureUnion([("word", word), ("char", char)])
    raise ValueError(f"Unknown TF-IDF feature kind: {kind}")

