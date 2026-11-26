import numpy as np
import pytest

from sonic_bridge.alignment import Alignment
from sonic_bridge.util import normalize


def test_recovers_known_rotation_and_roundtrips(tmp_path):
    rng = np.random.default_rng(42)
    x = normalize(rng.normal(size=(150, 8)))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    y = x @ q
    bridge = Alignment.fit(x, y, "procrustes")
    heldout = normalize(rng.normal(size=(30, 8)))
    np.testing.assert_allclose(bridge.transform(heldout), bridge.text_transform(heldout @ q), atol=1e-6)
    bridge.save(tmp_path / "map.npz")
    loaded = Alignment.load(tmp_path / "map.npz")
    np.testing.assert_allclose(loaded.transform(heldout), bridge.transform(heldout))


@pytest.mark.parametrize("rows,dimensions", [(20, 60), (100, 12)])
def test_ridge_handles_unequal_spaces_and_dual_solver(rows, dimensions):
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(rows, dimensions)), rng.normal(size=(rows, 7))
    bridge = Alignment.fit(x, y, "ridge", 0.1)
    xx, yy = normalize(x), normalize(y)
    xx, yy = xx - xx.mean(0), yy - yy.mean(0)
    expected = np.linalg.solve(xx.T @ xx + .1 * np.eye(dimensions), xx.T @ yy)
    np.testing.assert_allclose(bridge.weight, expected, atol=1e-5)
    assert bridge.transform(x).shape == (rows, 7)


def test_invalid_inputs():
    with pytest.raises(ValueError, match="equal dimensions"):
        Alignment.fit(np.eye(4), np.ones((4, 3)), "procrustes")
    with pytest.raises(ValueError, match="positive"):
        Alignment.fit(np.eye(4), np.eye(4), "ridge", 0)
    with pytest.raises(ValueError, match="non-finite"):
        Alignment.fit(np.array([[np.nan, 1], [1, 0]]), np.eye(2))


@pytest.mark.parametrize("method", ["cca", "rff"])
def test_shared_latent_recovery_and_portable_nonlinear_maps(tmp_path, method):
    rng = np.random.default_rng(17)
    latent = rng.normal(size=(120, 4))
    audio = latent @ rng.normal(size=(4, 9))
    text = latent @ rng.normal(size=(4, 7))
    model = Alignment.fit(audio[:100], text[:100], method, .01, components=4)
    model.save(tmp_path / "map.npz")
    loaded = Alignment.load(tmp_path / "map.npz")
    aa, tt = loaded.transform(audio[100:]), loaded.text_transform(text[100:])
    assert aa.shape == tt.shape
    assert np.isfinite(aa).all()
    np.testing.assert_allclose(aa, model.transform(audio[100:]))
    assert np.mean((aa @ tt.T).argmax(1) == np.arange(20)) > .5


def test_contrastive_multi_caption_projection_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    rng = np.random.default_rng(3)
    audio = normalize(rng.normal(size=(24, 6)))
    text = np.repeat(audio, 2, axis=0)
    bridge = Alignment.fit_contrastive(audio, text, np.repeat(np.arange(24), 2), epochs=50)
    bridge.save(tmp_path / "contrastive.npz")
    loaded = Alignment.load(tmp_path / "contrastive.npz")
    scores = loaded.transform(audio) @ loaded.text_transform(audio).T
    assert np.mean(scores.argmax(1) == np.arange(24)) > .8
