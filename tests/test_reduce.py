import numpy as np
import pytest
from sklearn.decomposition import PCA, TruncatedSVD

from embedding_lab.encode.reduce import Reducer

rng = np.random.default_rng(0)
# 300 vectors whose variance sits mostly in 3 directions of a 20-dimensional space.
BASIS = np.linalg.qr(rng.normal(size=(20, 20)))[0]
VECTORS = (rng.normal(size=(300, 20)) * np.r_[[10, 6, 3], np.full(17, 0.2)]) @ BASIS.T


def test_pca_projection_matches_sklearn_up_to_sign():
    ours = Reducer(3, "pca").fit(VECTORS).transform(VECTORS)
    theirs = PCA(n_components=3).fit_transform(VECTORS)
    theirs /= np.linalg.norm(theirs, axis=1, keepdims=True)
    assert np.allclose(np.abs(ours), np.abs(theirs), atol=1e-8)


def test_svd_projection_matches_truncated_svd_up_to_sign():
    ours = Reducer(3, "svd").fit(VECTORS).transform(VECTORS)
    theirs = TruncatedSVD(n_components=3, algorithm="arpack").fit_transform(VECTORS)
    theirs /= np.linalg.norm(theirs, axis=1, keepdims=True)
    assert np.allclose(np.abs(ours), np.abs(theirs), atol=1e-6)


def test_svd_at_full_size_keeps_every_cosine_and_centered_pca_does_not():
    shifted = VECTORS + 5.0  # a corpus whose vectors share a large common direction, as embeddings do
    unit = shifted / np.linalg.norm(shifted, axis=1, keepdims=True)
    cosine = unit @ unit.T
    svd = Reducer(20, "svd").fit(shifted).transform(shifted)
    pca = Reducer(20, "pca").fit(shifted).transform(shifted)
    assert np.allclose(svd @ svd.T, cosine, atol=1e-8)
    assert not np.allclose(pca @ pca.T, cosine, atol=1e-2)


def test_outputs_have_the_requested_size_and_unit_length():
    for method in ["pca", "svd", "truncate"]:
        out = Reducer(5, method).fit(VECTORS).transform(VECTORS)
        assert out.shape == (300, 5) and np.allclose(np.linalg.norm(out, axis=1), 1)


def test_pca_keeps_neighbours_better_than_truncation_when_variance_is_rotated():
    def top5(x):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        return np.argsort(-(x @ x.T), axis=1)[:, 1:6]

    full = top5(VECTORS)
    overlap = lambda reduced: np.mean([len(set(a) & set(b)) / 5 for a, b in zip(full, top5(reduced))])
    pca = overlap(Reducer(3, "pca").fit(VECTORS).transform(VECTORS))
    cut = overlap(Reducer(3, "truncate").transform(VECTORS))
    assert pca > cut + 0.2


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError, match="method"):
        Reducer(3, "random")
    with pytest.raises(ValueError, match="fit"):
        Reducer(3, "svd").transform(VECTORS)
