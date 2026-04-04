#!/usr/bin/env python3
import json
import os
import sys
import time
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
from src.ui.main_window import MainWindow
from src.utils.utils_detector import has_yasa


SAMPLE_EDF = ROOT / "SM_B_ave.edf"


def _process_events(app, cycles=20):
    for _ in range(cycles):
        app.processEvents()


def _build_hfo_midwindow_recording(out_dir):
    raw = mne.io.read_raw_edf(SAMPLE_EDF, preload=True, verbose=False)
    raw.crop(tmin=95, tmax=160)
    raw.pick(["POL IO3", "POL IO4", "POL IO5", "POL IO6", "POL IO7"])
    path = out_dir / "midwindow_hfo_smoke_raw.fif"
    raw.save(path, overwrite=True, verbose=False)
    return path


def _build_spindle_recording(out_dir):
    sfreq = 2000.0
    seconds = 30
    times = np.arange(int(sfreq * seconds)) / sfreq

    signal = np.random.default_rng(0).normal(scale=5e-7, size=times.shape)
    for start in [5, 12, 20]:
        mask = (times >= start) & (times < start + 1.0)
        envelope = np.hanning(mask.sum())
        signal[mask] += 2e-5 * np.sin(2 * np.pi * 13 * times[mask]) * envelope

    reference = np.random.default_rng(1).normal(scale=4e-7, size=times.shape)
    info = mne.create_info(["C3", "C4"], sfreq, ch_types="eeg")
    raw = mne.io.RawArray(np.vstack([signal, reference]), info, verbose=False)
    raw.info["line_freq"] = 60

    path = out_dir / "synthetic_spindle_raw_eeg.fif"
    raw.save(path, overwrite=True, verbose=False)
    return path


class DetectorClassifierAudit:
    def __init__(self):
        MainWindowModel.init_error_terminal_display = lambda self: None
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_root = ROOT / "artifacts" / "detector_classifier_audit" / self.timestamp
        self.input_dir = self.output_root / "inputs"
        self.screenshot_dir = self.output_root / "screenshots"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.steps = []
        self.summary = {
            "timestamp": self.timestamp,
            "paths": {},
            "environment": {
                "qt_qpa_platform": os.environ.get("QT_QPA_PLATFORM", ""),
                "has_yasa": has_yasa(),
            },
            "workflows": {},
        }

    def capture(self, widget, name, title, extra=None):
        _process_events(self.app, cycles=24)
        path = self.screenshot_dir / f"{name}.png"
        if not widget.grab().save(str(path)):
            raise RuntimeError(f"Failed to save screenshot: {path}")
        record = {
            "name": name,
            "title": title,
            "screenshot": str(path),
            "extra": extra or {},
        }
        self.steps.append(record)
        print(f"[capture] {title}: {path}")
        return record

    def wait_until(self, predicate, timeout_s=90.0, description="condition"):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            _process_events(self.app, cycles=10)
            if predicate():
                return
            time.sleep(0.05)
        raise TimeoutError(f"Timed out waiting for {description}")

    def create_window(self):
        window = MainWindow()
        window.show()
        _process_events(self.app, cycles=30)
        return window

    def load_recording(self, window, path):
        results = window.model.read_edf(str(path), None)
        window.model.update_edf_info(results)
        _process_events(self.app, cycles=40)

    def apply_detector_selection(self, window, detector_name):
        window.detector_mode_combo.setCurrentText(detector_name)
        _process_events(self.app, cycles=12)
        window.detector_apply_button.click()
        self.wait_until(
            lambda: getattr(getattr(window.model.backend, "param_detector", None), "detector_type", "").upper() == detector_name,
            description=f"{detector_name} detector apply",
        )

    def run_detector(self, window, detector_name):
        previous_runs = len(window.model.backend.get_run_summaries())
        self.apply_detector_selection(window, detector_name)
        window.detector_run_button.click()
        self.wait_until(
            lambda: len(window.model.backend.get_run_summaries()) == previous_runs + 1,
            timeout_s=120.0,
            description=f"{detector_name} detector run",
        )
        self.wait_until(
            lambda: window.detector_run_button.isEnabled(),
            description=f"{detector_name} detector button reset",
        )
        _process_events(self.app, cycles=24)

    def apply_classifier_selection(self, window, mode_name, *, custom_device=None):
        window.classifier_mode_combo.setCurrentText(mode_name)
        _process_events(self.app, cycles=12)
        if mode_name == "Custom":
            if custom_device is not None:
                window.classifier_device_input.setText(custom_device)
            window.classifier_apply_button.click()
            self.wait_until(
                lambda: getattr(getattr(window.model.backend, "param_classifier", None), "device", "") == (custom_device or ""),
                description="custom classifier apply",
            )
        else:
            self.wait_until(
                lambda: window.classifier_mode_combo.currentText() == mode_name,
                description=f"{mode_name} classifier selection",
            )
        _process_events(self.app, cycles=12)

    def run_classifier(self, window, mode_name, *, use_spike=True, use_ehfo=True):
        self.apply_classifier_selection(window, mode_name)
        if hasattr(window, "use_spike_checkbox"):
            window.use_spike_checkbox.setChecked(use_spike)
        if hasattr(window, "overview_use_spike_checkbox"):
            window.overview_use_spike_checkbox.setChecked(use_spike)
        if hasattr(window, "use_ehfo_checkbox"):
            window.use_ehfo_checkbox.setChecked(use_ehfo)
        if hasattr(window, "overview_use_ehfo_checkbox"):
            window.overview_use_ehfo_checkbox.setChecked(use_ehfo)
        window.classifier_run_button.click()
        self.wait_until(
            lambda: bool(getattr(window.model.backend, "classified", False)),
            timeout_s=120.0,
            description=f"{mode_name} classifier run",
        )
        self.wait_until(
            lambda: window.classifier_run_button.isEnabled(),
            description=f"{mode_name} classifier button reset",
        )
        _process_events(self.app, cycles=24)

    def switch_workflow(self, window, biomarker_type):
        action_map = {
            "HFO": window.new_hfo_run_action,
            "Spindle": window.new_spindle_run_action,
            "Spike": window.new_spike_run_action,
        }
        action_map[biomarker_type].trigger()
        self.wait_until(
            lambda: window.combo_box_biomarker.currentText() == biomarker_type and getattr(window.model.backend, "eeg_data", None) is not None,
            description=f"{biomarker_type} workflow load",
        )
        _process_events(self.app, cycles=30)

    def audit_hfo(self, window, hfo_path):
        self.load_recording(window, hfo_path)
        hfo_summary = {
            "detector_items": [window.detector_mode_combo.itemText(i) for i in range(window.detector_mode_combo.count())],
            "classifier_items": [window.classifier_mode_combo.itemText(i) for i in range(window.classifier_mode_combo.count())],
            "detectors": {},
            "classifiers": {},
        }
        self.capture(
            window,
            "01_hfo_loaded",
            "HFO Workflow Loaded",
            {
                "active_selector": window.active_run_selector.currentText(),
                "run_stats_enabled": window.compare_runs_button.isEnabled(),
            },
        )

        for detector_name in ["STE", "MNI", "HIL"]:
            self.apply_detector_selection(window, detector_name)
            detector_state = {
                "selected": window.detector_mode_combo.currentText(),
                "run_button": window.detector_run_button.text(),
                "param_detector": getattr(window.model.backend.param_detector, "detector_type", ""),
            }
            hfo_summary["detectors"][detector_name] = {"apply": detector_state}
            self.capture(window, f"detector_apply_{detector_name.lower()}", f"HFO Detector Applied: {detector_name}", detector_state)

        for detector_name in ["STE", "MNI", "HIL"]:
            self.run_detector(window, detector_name)
            current_run = window.model.backend.analysis_session.get_active_run()
            run_state = {
                "active_run": current_run.detector_name if current_run is not None else None,
                "run_count": len(window.model.backend.get_run_summaries()),
                "event_count": int(window.model.backend.event_features.get_num_biomarker()) if window.model.backend.event_features is not None else 0,
                "run_stats_enabled": window.compare_runs_button.isEnabled(),
            }
            hfo_summary["detectors"][detector_name]["run"] = run_state
            self.capture(window, f"detector_run_{detector_name.lower()}", f"HFO Detector Run: {detector_name}", run_state)

        self.run_classifier(window, "Hugging Face CPU", use_spike=True, use_ehfo=False)
        classifier_param = window.model.backend.get_classifier_param()
        hfo_summary["classifiers"]["Hugging Face CPU"] = {
            "classified": window.model.backend.classified,
            "device": getattr(classifier_param, "device", ""),
            "model_type": getattr(classifier_param, "model_type", ""),
        }
        self.capture(
            window,
            "classifier_run_hf_cpu",
            "HFO Classifier Run: Hugging Face CPU",
            hfo_summary["classifiers"]["Hugging Face CPU"],
        )

        self.apply_classifier_selection(window, "Hugging Face GPU")
        gpu_param = window.model.backend.get_classifier_param()
        hfo_summary["classifiers"]["Hugging Face GPU"] = {
            "device": getattr(gpu_param, "device", ""),
            "model_type": getattr(gpu_param, "model_type", ""),
        }
        self.capture(
            window,
            "classifier_select_hf_gpu",
            "HFO Classifier Selected: Hugging Face GPU",
            hfo_summary["classifiers"]["Hugging Face GPU"],
        )

        self.apply_classifier_selection(window, "Custom", custom_device="cpu")
        custom_param = window.model.backend.get_classifier_param()
        hfo_summary["classifiers"]["Custom"] = {
            "device": getattr(custom_param, "device", ""),
            "model_type": getattr(custom_param, "model_type", ""),
        }
        self.capture(
            window,
            "classifier_apply_custom",
            "HFO Classifier Applied: Custom",
            hfo_summary["classifiers"]["Custom"],
        )

        window.compare_runs_button.click()
        self.wait_until(lambda: hasattr(window, "run_stats_dialog") and window.run_stats_dialog.isVisible(), description="run stats dialog visible")
        self.capture(
            window.run_stats_dialog,
            "run_stats_hfo",
            "HFO Run Statistics",
            {
                "rows": window.run_table.rowCount(),
                "comparison_rows": window.comparison_table.rowCount(),
            },
        )
        window.run_stats_dialog.close()
        _process_events(self.app, cycles=12)
        self.summary["workflows"]["HFO"] = hfo_summary

    def audit_spindle(self, window, spindle_path):
        window.model.current_recording_path = str(spindle_path)
        self.switch_workflow(window, "Spindle")
        spindle_summary = {
            "detector_items": [window.detector_mode_combo.itemText(i) for i in range(window.detector_mode_combo.count())],
            "classifier_items": [window.classifier_mode_combo.itemText(i) for i in range(window.classifier_mode_combo.count())],
            "has_yasa": has_yasa(),
        }
        self.capture(
            window,
            "spindle_loaded",
            "Spindle Workflow Loaded",
            {
                "detector_items": spindle_summary["detector_items"],
                "classifier_items": spindle_summary["classifier_items"],
            },
        )

        window.detector_apply_button.click()
        _process_events(self.app, cycles=12)
        spindle_summary["detector_apply"] = getattr(window.model.backend.param_detector, "detector_type", "")

        for classifier_mode in ["Hugging Face CPU", "Hugging Face GPU", "Custom"]:
            if classifier_mode == "Custom":
                self.apply_classifier_selection(window, classifier_mode, custom_device="cpu")
            else:
                self.apply_classifier_selection(window, classifier_mode)
            spindle_summary.setdefault("classifier_apply", {})[classifier_mode] = getattr(
                window.model.backend.get_classifier_param(), "model_type", ""
            )

        if has_yasa():
            self.run_detector(window, "YASA")
            spindle_summary["detector_run"] = {
                "run_count": len(window.model.backend.get_run_summaries()),
                "event_count": int(window.model.backend.event_features.get_num_biomarker()) if window.model.backend.event_features is not None else 0,
            }
            self.run_classifier(window, "Hugging Face CPU", use_spike=True, use_ehfo=False)
            spindle_summary["classifier_run"] = {
                "classified": window.model.backend.classified,
                "device": getattr(window.model.backend.get_classifier_param(), "device", ""),
            }
        else:
            spindle_summary["detector_run"] = {"skipped": "Optional yasa dependency is unavailable."}
            spindle_summary["classifier_run"] = {"skipped": "Detection was not available without yasa."}

        self.capture(window, "spindle_detector_classifier", "Spindle Detector and Classifier Audit", spindle_summary)
        self.summary["workflows"]["Spindle"] = spindle_summary

    def audit_spike(self, window):
        self.switch_workflow(window, "Spike")
        spike_summary = {
            "detector_items": [window.detector_mode_combo.itemText(i) for i in range(window.detector_mode_combo.count())],
            "classifier_items": [window.classifier_mode_combo.itemText(i) for i in range(window.classifier_mode_combo.count())],
            "detector_run_enabled": window.detector_run_button.isEnabled(),
            "classifier_run_enabled": window.classifier_run_button.isEnabled(),
            "detector_apply_enabled": window.detector_apply_button.isEnabled(),
        }
        self.capture(window, "spike_review_mode", "Spike Review Workflow", spike_summary)
        self.summary["workflows"]["Spike"] = spike_summary

    def write_report(self):
        self.summary["steps"] = self.steps
        summary_path = self.output_root / "summary.json"
        summary_path.write_text(json.dumps(self.summary, indent=2), encoding="utf-8")

        lines = [
            "# Main Detector + Classifier Audit",
            "",
            f"- Timestamp: `{self.timestamp}`",
            f"- HFO detector items: `{', '.join(self.summary['workflows']['HFO']['detector_items'])}`",
            f"- HFO classifier items: `{', '.join(self.summary['workflows']['HFO']['classifier_items'])}`",
            f"- Spindle detector items: `{', '.join(self.summary['workflows']['Spindle']['detector_items'])}`",
            f"- Spindle classifier items: `{', '.join(self.summary['workflows']['Spindle']['classifier_items'])}`",
            f"- Spike detector items: `{', '.join(self.summary['workflows']['Spike']['detector_items'])}`",
            f"- Spike classifier items: `{', '.join(self.summary['workflows']['Spike']['classifier_items'])}`",
            "",
            "## Steps",
            "",
        ]
        for step in self.steps:
            lines.append(f"- {step['title']}: `{Path(step['screenshot']).name}`")
        report_path = self.output_root / "report.md"
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[report] {report_path}")
        print(f"[summary] {summary_path}")
        return report_path, summary_path

    def run(self):
        hfo_path = _build_hfo_midwindow_recording(self.input_dir)
        spindle_path = _build_spindle_recording(self.input_dir)
        self.summary["paths"]["hfo_input"] = str(hfo_path)
        self.summary["paths"]["spindle_input"] = str(spindle_path)

        window = self.create_window()
        try:
            self.audit_hfo(window, hfo_path)
            self.audit_spindle(window, spindle_path)
            self.audit_spike(window)
        finally:
            window.close()
            _process_events(self.app, cycles=24)
        return self.write_report()


if __name__ == "__main__":
    audit = DetectorClassifierAudit()
    audit.run()
