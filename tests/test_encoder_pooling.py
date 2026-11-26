from types import SimpleNamespace

import numpy as np
import pytest

from sonic_bridge.encoders import JepaTextEncoder
from sonic_bridge.util import normalize


def test_frame_pooling_ignores_padding_and_keeps_scene_behavior():
    torch = pytest.importorskip("torch")
    encoder = JepaTextEncoder.__new__(JepaTextEncoder)
    encoder.torch, encoder.device = torch, "cpu"
    encoder.sample_rate = 16000
    encoder.audio_processor = lambda *args, **kwargs: {"input_values": torch.zeros(1, 10)}
    outputs = SimpleNamespace(pooler_output=torch.tensor([[0., 1.]]),
                              last_hidden_state=torch.tensor([[[2., 0.], [1., 1.], [999., 999.]]]),
                              attention_mask=torch.tensor([[1, 1, 0]]))
    encoder.audio_model = lambda **kwargs: outputs
    for pooling, expected in [("scene", [[0, 1]]), ("frame_mean", [[1.5, .5]]), ("frame_max", [[2, 1]])]:
        encoder.audio_pooling = pooling
        np.testing.assert_allclose(encoder.encode_audio([np.zeros(10)]), normalize(np.array(expected)), atol=1e-6)
