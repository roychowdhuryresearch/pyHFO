import logging

from src.hfo_app import HFO_App
from src.classifer import _silence_hf_public_download_notice
from src.param.param_classifier import ParamClassifier


def test_param_classifier_round_trip_preserves_source_preference():
    classifier_param = ParamClassifier(
        artifact_path="/tmp/model_a.tar",
        artifact_card="roychowdhuryresearch/HFO-artifact",
        model_type="default_cpu",
        source_preference="huggingface",
    )

    restored = ParamClassifier.from_dict(classifier_param.to_dict())

    assert restored.source_preference == "huggingface"
    assert restored.artifact_card == "roychowdhuryresearch/HFO-artifact"


def test_default_classifier_presets_prefer_hugging_face():
    app = HFO_App()

    app.set_default_cpu_classifier()

    assert app.param_classifier.source_preference == "huggingface"
    assert app.param_classifier.artifact_card == "roychowdhuryresearch/HFO-artifact"


def test_silence_hf_public_download_notice_restores_logger_level():
    logger = logging.getLogger("huggingface_hub.utils._http")
    previous_level = logger.level
    logger.setLevel(logging.WARNING)

    with _silence_hf_public_download_notice():
        assert logger.level == logging.ERROR

    assert logger.level == logging.WARNING
    logger.setLevel(previous_level)
