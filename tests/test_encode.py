import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

from embedding_lab.encode.cache import EmbeddingCache
from embedding_lab.encode.encoder import MINILM, Encoder

TEXTS = ["Statins and breast cancer survival.", "A short one.",
         "Dietary fibre intake was measured in a cohort of adults over ten years, with mixed results."]


def test_mean_pooling_matches_sentence_transformers():
    ours = Encoder(model_name=MINILM, pooling="mean", normalize=True, device="cpu").encode(TEXTS)
    theirs = SentenceTransformer(MINILM, device="cpu").encode(TEXTS, normalize_embeddings=True)
    assert ours.shape == (3, 384)
    assert np.allclose(ours, theirs, atol=1e-4)


def test_cls_pooling_differs_from_mean_and_normalizes():
    cls = Encoder(pooling="cls", device="cpu").encode(TEXTS)
    mean = Encoder(pooling="mean", device="cpu").encode(TEXTS)
    assert not np.allclose(cls, mean, atol=1e-2)
    assert np.allclose(np.linalg.norm(cls, axis=1), 1.0, atol=1e-5)


def test_unnormalized_vectors_have_varied_lengths():
    raw = Encoder(normalize=False, device="cpu").encode(TEXTS)
    assert np.ptp(np.linalg.norm(raw, axis=1)) > 0.01


def test_bad_pooling_is_rejected():
    with pytest.raises(ValueError, match="pooling"):
        Encoder(pooling="max")


def test_cache_returns_the_saved_embeddings_and_keys_on_settings(tmp_path):
    cache = EmbeddingCache(directory=tmp_path)
    encoder = Encoder(device="cpu")
    first = cache.encode(encoder, "t", TEXTS)
    assert len(list(tmp_path.glob("t-*.npy"))) == 1
    assert np.array_equal(cache.encode(encoder, "t", TEXTS), first)
    cache.encode(Encoder(pooling="cls", device="cpu"), "t", TEXTS)
    assert len(list(tmp_path.glob("t-*.npy"))) == 2
