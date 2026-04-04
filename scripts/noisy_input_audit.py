#!/usr/bin/env python3
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import mne
import numpy as np
from PyQt5 import QtWidgets

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.main_window_model import MainWindowModel
from src.param.param_classifier import ParamClassifier
from src.ui.main_window import MainWindow
from src.ui.quick_detection import HFOQuickDetector


def _process_events(app, cycles=12):
    for _ in range(cycles):
        app.processEvents()


def _build_tiny_recording():
    sfreq = 200.0
    samples = int(sfreq * 2)
    times = np.arange(samples) / sfreq
    info = mne.create_info(["B10Ref", "A2Ref", "A1Ref"], sfreq, ch_types="eeg")
    data = np.vstack(
        [
            np.sin(2 * np.pi * 8 * times),
            np.cos(2 * np.pi * 12 * times) * 0.8,
            np.sin(2 * np.pi * 16 * times) * 0.5,
        ]
    ) * 1e-6
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.info["line_freq"] = 60
    path = Path(tempfile.mkdtemp()) / "tiny_raw.fif"
    raw.save(path, overwrite=True, verbose=False)
    return path


@contextmanager
def _suppressed_message_boxes():
    original_exec = QtWidgets.QMessageBox.exec_
    original_critical = QtWidgets.QMessageBox.critical
    original_information = QtWidgets.QMessageBox.information
    original_warning = QtWidgets.QMessageBox.warning
    QtWidgets.QMessageBox.exec_ = lambda self: QtWidgets.QMessageBox.Ok
    QtWidgets.QMessageBox.critical = lambda *args, **kwargs: QtWidgets.QMessageBox.Ok
    QtWidgets.QMessageBox.information = lambda *args, **kwargs: QtWidgets.QMessageBox.Ok
    QtWidgets.QMessageBox.warning = lambda *args, **kwargs: QtWidgets.QMessageBox.Ok
    try:
        yield
    finally:
        QtWidgets.QMessageBox.exec_ = original_exec
        QtWidgets.QMessageBox.critical = original_critical
        QtWidgets.QMessageBox.information = original_information
        QtWidgets.QMessageBox.warning = original_warning


class _QuickDetectionBackend:
    def __init__(self):
        self.n_jobs = 1
        self._classifier_param = ParamClassifier(
            artifact_path="artifact.tar",
            spike_path="spike.tar",
            ehfo_path="ehfo.tar",
            use_spike=True,
            use_ehfo=True,
            device="cpu",
            batch_size=32,
            model_type="default_cpu",
            source_preference="auto",
        )

    def get_classifier_param(self):
        return self._classifier_param

    def set_default_cpu_classifier(self):
        self._classifier_param.device = "cpu"
        self._classifier_param.model_type = "default_cpu"

    def set_default_gpu_classifier(self):
        self._classifier_param.device = "cuda:0"
        self._classifier_param.model_type = "default_gpu"


def run_quick_detection_noise_checks(app):
    dialog = HFOQuickDetector(backend=_QuickDetectionBackend())
    results = {}
    try:
        dialog.show()
        _process_events(app)
        dialog.fname = "/tmp/noisy_sample_raw.fif"
        dialog.filename = "noisy_sample_raw.fif"
        dialog.update_edf_info(["noisy_sample_raw.fif", "2000", "4", "10000"])
        dialog.detectionTypeComboBox.setCurrentText("MNI")
        dialog.qd_use_classifier_checkbox.setChecked(True)

        dialog.qd_mni_epoch_time_input.setText("nan")
        try:
            dialog.collect_run_configuration()
            results["nan_detector_value"] = {"ok": False, "reason": "accepted nan unexpectedly"}
        except ValueError as exc:
            results["nan_detector_value"] = {"ok": True, "message": str(exc)}

        dialog.init_default_mni_input_params()
        dialog.qd_classifier_device_input.setText("gpu??")
        try:
            dialog.get_classifier_param()
            results["invalid_classifier_device"] = {"ok": False, "reason": "accepted invalid device unexpectedly"}
        except ValueError as exc:
            results["invalid_classifier_device"] = {"ok": True, "message": str(exc)}

        dialog.qd_classifier_device_input.setText("cpu")
        dialog.qd_classifier_batch_size_input.setText("0")
        try:
            dialog.get_classifier_param()
            results["zero_batch_size"] = {"ok": False, "reason": "accepted zero batch unexpectedly"}
        except ValueError as exc:
            results["zero_batch_size"] = {"ok": True, "message": str(exc)}
    finally:
        dialog.close()
        _process_events(app)
    return results


def run_main_window_noise_checks(app, recording_path):
    MainWindowModel.init_error_terminal_display = lambda self: None
    window = MainWindow()
    results = {}
    try:
        window.show()
        _process_events(app)
        data = window.model.read_edf(str(recording_path), None)
        window.model.update_edf_info(data)
        _process_events(app, cycles=24)

        with _suppressed_message_boxes():
            window.detector_mode_combo.setCurrentText("MNI")
            _process_events(app)
            results["valid_mni_apply"] = {
                "ok": window.model.apply_selected_detector_parameters(show_feedback=False),
                "detector": getattr(window.model.backend.param_detector, "detector_type", None),
            }

            window.mni_epoch_time_input.setText("nan")
            _process_events(app)
            before_runs = len(window.model.backend.get_run_summaries())
            apply_ok = window.model.apply_selected_detector_parameters(show_feedback=False)
            window.model.run_selected_detector_workflow()
            after_runs = len(window.model.backend.get_run_summaries())
            results["invalid_mni_apply_after_valid"] = {
                "ok": apply_ok is False and window.mni_detect_button.isEnabled() is False,
                "apply_ok": apply_ok,
                "detect_button_enabled": window.mni_detect_button.isEnabled(),
                "run_count_unchanged": before_runs == after_runs,
            }

            window.fp_input.setText("inf")
            _process_events(app)
            previous_filter = getattr(window.model.backend, "param_filter", None)
            window.model.filter_data()
            _process_events(app)
            results["invalid_filter_value"] = {
                "ok": getattr(window.model.backend, "param_filter", None) is previous_filter,
                "filter_param_changed": getattr(window.model.backend, "param_filter", None) is not previous_filter,
            }

            window.classifier_mode_combo.setCurrentText("Custom")
            _process_events(app)
            baseline_classifier = window.model.backend.get_classifier_param()
            baseline_device = baseline_classifier.device
            baseline_model_type = baseline_classifier.model_type

            window.classifier_device_input.setText("gpu??")
            invalid_device_ok = window.model.apply_classifier_setup()
            current_classifier = window.model.backend.get_classifier_param()
            results["invalid_custom_classifier_device"] = {
                "ok": invalid_device_ok is False,
                "apply_ok": invalid_device_ok,
                "device_preserved": current_classifier.device == baseline_device,
                "model_type_preserved": current_classifier.model_type == baseline_model_type,
            }

            window.classifier_device_input.setText("cpu")
            window.classifier_artifact_filename.setText("")
            window.classifier_artifact_card_name.setText("")
            missing_source_ok = window.model.apply_classifier_setup()
            results["missing_custom_artifact_source"] = {
                "ok": missing_source_ok is False,
                "apply_ok": missing_source_ok,
            }
    finally:
        window.close()
        _process_events(app)
    return results


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = ROOT / "artifacts" / "noisy_input_audit" / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    tiny_recording = _build_tiny_recording()
    summary = {
        "timestamp": timestamp,
        "recording": str(tiny_recording),
        "quick_detection": run_quick_detection_noise_checks(app),
        "main_window": run_main_window_noise_checks(app, tiny_recording),
    }

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
