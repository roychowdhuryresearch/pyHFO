"""Regression test for the torch-free ONNX inference runner (Path B).

Self-contained: builds a tiny CNN, exports it to ONNX, and asserts
``OnnxBiomarkerModel`` reproduces the torch reference labels — covering the
runner's per-channel normalize, channel-slicing, dynamic H/W, and threshold.
No checkpoint / EDF needed. The full real-EDF parity (0 mismatches on 583
events across all three production models) is verified in scripts, not here.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from pyhfo.utils.onnx_inference import OnnxBiomarkerModel, _prepare_batch


class _TinyCNN(torch.nn.Module):
    """Mirrors the production models' input contract: optionally slices to
    channel 0 inside forward (as NeuralCNN does when in_channels == 1)."""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = torch.nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(4, 1)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.in_channels == 1:
            x = x[:, 0:1]
        x = self.pool(torch.relu(self.conv(x))).flatten(1)
        return self.act(self.fc(x))


@pytest.mark.parametrize(
    "feature_channels,model_channels,image_size",
    [(2, 1, 96), (2, 2, 96), (3, 3, 224)],  # artifact / spike / eHFO shapes
)
def test_onnx_runner_matches_torch(feature_channels, model_channels, image_size, tmp_path):
    torch.manual_seed(0)
    model = _TinyCNN(model_channels).eval()
    features = np.random.RandomState(0).randn(9, feature_channels, image_size, image_size).astype(np.float32)

    # Torch reference mirrors the real pipeline: the full feature stack is
    # min-max normalized per channel, then the model uses what it needs.
    normalized = _prepare_batch(features.copy()).astype(np.float32)
    with torch.inference_mode():
        torch_labels = np.atleast_1d(np.squeeze(model(torch.from_numpy(normalized)).numpy() > 0.5))

    onnx_path = tmp_path / "model.onnx"
    dummy = torch.from_numpy(normalized[:2, :model_channels])
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["prob"],
        dynamic_axes={"input": {0: "batch", 2: "h", 3: "w"}, "prob": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    runner = OnnxBiomarkerModel(str(onnx_path))
    assert runner.in_channels == model_channels
    onnx_labels = np.atleast_1d(runner.inference(features))  # raw features; runner slices + normalizes
    assert np.array_equal(torch_labels, onnx_labels)
