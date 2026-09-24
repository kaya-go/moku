"""What `moku publish` pushes must be enough to re-export the ONNX Kaya runs."""

import torch
from transformers import RTDetrConfig, RTDetrForObjectDetection

from moku.corner_head import attach_corner_head
from moku.hub import load_for_publish


def test_load_for_publish_keeps_the_corner_head(tmp_path):
    torch.manual_seed(0)
    model = RTDetrForObjectDetection(RTDetrConfig(num_labels=3))
    head = attach_corner_head(model, hidden=16)
    model.save_pretrained(tmp_path)

    loaded = load_for_publish(str(tmp_path))

    state = loaded.state_dict()
    for name, weight in head.state_dict().items():
        assert torch.equal(state[f"corner_head.{name}"], weight)

    # Saved again (what push_to_hub does), the head survives the round trip.
    loaded.save_pretrained(tmp_path / "again")
    assert load_for_publish(str(tmp_path / "again")).corner_head is not None
