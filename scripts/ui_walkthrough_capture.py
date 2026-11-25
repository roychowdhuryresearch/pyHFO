#!/usr/bin/env python3
import json
import os
import sys
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
from src.param.param_detector import ParamDetector, ParamYASA
from src.param.param_filter import ParamFilterSpindle
from src.spindle_app import SpindleApp
from src.ui.annotation import Annotation
from src.ui.main_window import MainWindow
from src.ui.quick_detection import HFOQuickDetector
from src.utils.utils_detector import has_yasa


SAMPLE_EDF = ROOT / "SM_B_ave.edf"

MAIN_WINDOW_CAPTURE_SIZE = (1600, 980)
QUICK_DETECTION_CAPTURE_SIZE = (1380, 1180)
RUN_STATS_CAPTURE_SIZE = (1480, 920)
ANNOTATION_CAPTURE_SIZE = (1760, 1180)


def _process_events(app, cycles=25):
    for _ in range(cycles):
        app.processEvents()


def _close_widget(widget, app):
    if widget is None:
        return
    widget.close()
    _process_events(app)


def _resize_widget(widget, size, app, cycles=30):
    if widget is None:
        return
    width, height = size
    widget.resize(int(width), int(height))
    widget.adjustSize()
    widget.resize(int(width), int(height))
    _process_events(app, cycles=cycles)


def _load_recording_into_model(model, path):
    results = model.read_edf(str(path), None)
    model.update_edf_info(results)


def _run_filter_via_model(model):
    model._filter(None)
    model.filtering_complete()


def _run_hfo_detector_via_model(model, detector_name):
    model.window.detector_mode_combo.setCurrentText(detector_name)
    if detector_name == "STE":
        model.save_ste_params()
    elif detector_name == "MNI":
        model.save_mni_params()
    else:
        raise ValueError(f"Unsupported HFO detector for walkthrough: {detector_name}")
    model.backend.detect_biomarker()
    model._detect_finished()
    return model.backend.analysis_session.active_run_id


def _run_spindle_detector_via_model(model):
    model.window.detector_mode_combo.setCurrentText("YASA")
    model.save_yasa_params()
    model.backend.detect_biomarker()
    model._detect_finished()
    return model.backend.analysis_session.active_run_id


def _run_classifier_via_model(model, use_spike=True, use_ehfo=True):
    model.window.use_spike_checkbox.setChecked(use_spike)
    model.window.overview_use_spike_checkbox.setChecked(use_spike)
    if hasattr(model.window, "use_ehfo_checkbox"):
        model.window.use_ehfo_checkbox.setChecked(use_ehfo)
    if hasattr(model.window, "overview_use_ehfo_checkbox"):
        model.window.overview_use_ehfo_checkbox.setChecked(use_ehfo)
    model._classify()
    model._classify_finished()


def _find_run_row(window, detector_name):
    for row in range(window.run_table.rowCount()):
        item = window.run_table.item(row, 2)
        if item is not None and item.text() == detector_name:
            return row
    raise AssertionError(f"Could not find detector row '{detector_name}'")


def _find_annotation_window(parent_window):
    candidates = parent_window.findChildren(Annotation)
    if not candidates:
        raise AssertionError("Annotation window was not created")
    return candidates[-1]


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


@contextmanager
def _patched_save_dialog(path):
    original = QtWidgets.QFileDialog.getSaveFileName
    QtWidgets.QFileDialog.getSaveFileName = lambda *args, **kwargs: (str(path), "")
    try:
        yield
    finally:
        QtWidgets.QFileDialog.getSaveFileName = original


class UIWalkthroughCapture:
    def __init__(self):
        MainWindowModel.init_error_terminal_display = lambda self: None
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_root = ROOT / "artifacts" / "ui_walkthrough" / self.timestamp
        self.screenshot_dir = self.output_root / "screenshots"
        self.export_dir = self.output_root / "exports"
        self.input_dir = self.output_root / "inputs"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.steps = []

    def capture(self, widget, name, title, note="", extra=None):
        _process_events(self.app)
        path = self.screenshot_dir / f"{name}.png"
        if not widget.grab().save(str(path)):
            raise RuntimeError(f"Failed to save screenshot: {path}")
        self.steps.append(
            {
                "name": name,
                "title": title,
                "note": note,
                "screenshot": path,
                "extra": extra or {},
            }
        )
        print(f"[capture] {title}: {path}")
        return path

    def create_main_window(self):
        window = MainWindow()
        _resize_widget(window, MAIN_WINDOW_CAPTURE_SIZE, self.app)
        window.show()
        _process_events(self.app)
        return window

    def run_quick_detection_flow(self, hfo_input_path):
        dialog = HFOQuickDetector()
        _resize_widget(dialog, QUICK_DETECTION_CAPTURE_SIZE, self.app)
        dialog.show()
        _process_events(self.app)
        try:
            results = dialog.read_edf(str(hfo_input_path), None)
            dialog.update_edf_info(results)
            self.capture(
                dialog,
                "01_quick_detection_loaded",
                "Quick Detection Loaded",
                "Loaded a mid-window HFO sample with real POL IO channels.",
                {
                    "recording": str(hfo_input_path),
                    "sample_freq": dialog.qd_sampfreq.text(),
                    "channels": dialog.qd_numchannels.text(),
                },
            )

            dialog.detectionTypeComboBox.setCurrentText("MNI")
            dialog.update_detector_tab("MNI")
            dialog.qd_use_classifier_checkbox.setChecked(True)
            dialog.qd_use_spikes_checkbox.setChecked(True)
            dialog.qd_use_ehfo_checkbox.setChecked(True)
            dialog.qd_ignore_sec_before_input.setText("0")
            dialog.qd_ignore_sec_after_input.setText("0")
            dialog.qd_excel_checkbox.setChecked(True)
            dialog.qd_npz_checkbox.setChecked(True)
            self.capture(
                dialog,
                "02_quick_detection_configured",
                "Quick Detection Configured",
                "Configured MNI detection with artifact, spike, and eHFO classification plus Excel and session export.",
            )

            run_config = dialog.collect_run_configuration()
            outputs = dialog._run(run_config, None)
            dialog._run_finished(outputs)
            dialog._run_cleanup()
            event_features = dialog.backend.event_features
            assert event_features is not None and event_features.get_num_biomarker() > 0
            assert event_features.artifact_predicted is True
            assert event_features.spike_predicted is True
            assert event_features.ehfo_predicted is True
            assert Path(outputs["excel_output"]).exists()
            assert Path(outputs["npz_output"]).exists()
            self.capture(
                dialog,
                "03_quick_detection_complete",
                "Quick Detection Complete",
                "Quick detection completed with real MNI events and three classifier heads.",
                {
                    "num_events": int(event_features.get_num_biomarker()),
                    "excel_output": outputs["excel_output"],
                    "npz_output": outputs["npz_output"],
                },
            )
            return outputs
        finally:
            _close_widget(dialog, self.app)

    def run_hfo_main_flow(self, hfo_input_path):
        window = self.create_main_window()
        run_stats_dialog = None
        annotation_window = None
        try:
            self.capture(
                window,
                "10_hfo_home",
                "HFO Workspace Home",
                "Fresh app launch in the default HFO workspace.",
            )

            _load_recording_into_model(window.model, hfo_input_path)
            self.capture(
                window,
                "11_hfo_loaded",
                "HFO Recording Loaded",
                "Loaded the same HFO mid-window sample into the main workspace.",
                {
                    "recording": str(hfo_input_path),
                    "sample_freq": window.main_sampfreq.text(),
                    "channels": window.main_numchannels.text(),
                },
            )

            _run_filter_via_model(window.model)
            self.capture(
                window,
                "12_hfo_filtered",
                "HFO Filtered",
                "Applied the overview filter and enabled the filtered waveform state.",
            )

            ste_run_id = _run_hfo_detector_via_model(window.model, "STE")
            ste_events = int(window.model.backend.event_features.get_num_biomarker())
            assert ste_events > 0
            self.capture(
                window,
                "13_hfo_ste_detected",
                "HFO STE Run",
                "Created the first HFO run with STE.",
                {"ste_run_id": ste_run_id, "num_events": ste_events},
            )

            mni_run_id = _run_hfo_detector_via_model(window.model, "MNI")
            mni_events = int(window.model.backend.event_features.get_num_biomarker())
            assert mni_run_id != ste_run_id
            assert mni_events > 0
            self.capture(
                window,
                "14_hfo_mni_detected",
                "HFO MNI Run",
                "Created the second HFO run with MNI so run comparison and switching are available.",
                {
                    "mni_run_id": mni_run_id,
                    "num_events": mni_events,
                    "run_count": len(window.model.backend.get_run_summaries()),
                },
            )

            window.model.set_classifier_param_cpu_default()
            _run_classifier_via_model(window.model, use_spike=True, use_ehfo=True)
            event_features = window.model.backend.event_features
            assert event_features.artifact_predicted is True
            assert event_features.spike_predicted is True
            assert event_features.ehfo_predicted is True
            self.capture(
                window,
                "15_hfo_classified",
                "HFO Classified",
                "Ran artifact, spkHFO, and eHFO classification for the active MNI run.",
                {
                    "num_events": int(event_features.get_num_biomarker()),
                    "artifacts": int(event_features.get_num_artifact()),
                    "real": int(event_features.get_num_real()),
                    "spike": int(event_features.get_num_spike()),
                    "ehfo": int(event_features.get_num_ehfo()),
                },
            )

            excel_path = self.export_dir / "hfo_clinical_summary.xlsx"
            report_path = self.export_dir / "hfo_analysis_report.html"
            with _patched_save_dialog(excel_path):
                window.model.save_to_excel()
            with _patched_save_dialog(report_path):
                window.model.save_analysis_report()
            assert excel_path.exists()
            assert report_path.exists()

            window.model.show_run_comparison()
            _process_events(self.app)
            run_stats_dialog = getattr(window, "run_stats_dialog", None)
            if run_stats_dialog is not None:
                _resize_widget(run_stats_dialog, RUN_STATS_CAPTURE_SIZE, self.app)
                self.capture(
                    run_stats_dialog,
                    "16_hfo_run_comparison",
                    "HFO Run Comparison",
                    "Opened the run comparison dialog to inspect agreement between STE and MNI.",
                    {
                        "excel_export": str(excel_path),
                        "report_export": str(report_path),
                    },
                )

            ste_row = _find_run_row(window, "STE")
            window.model.activate_run_from_table(ste_row, 0)
            _process_events(self.app)
            window.model.accept_active_run()
            _process_events(self.app)
            assert window.model.backend.get_decision_summary()["accepted_detector"] == "STE"
            self.capture(
                window,
                "17_hfo_run_switched",
                "HFO Run Switched And Accepted",
                "Activated the STE run from the run table and marked it as the accepted downstream run.",
            )

            window.model.open_annotation()
            _process_events(self.app)
            annotation_window = _find_annotation_window(window)
            _resize_widget(annotation_window, ANNOTATION_CAPTURE_SIZE, self.app)
            self.capture(
                annotation_window,
                "18_hfo_annotation_open",
                "HFO Annotation Open",
                "Opened the event-by-event review window for the active STE run.",
            )

            annotation_window.select_annotation_option("Artifact")
            self.capture(
                annotation_window,
                "19_hfo_annotation_selected",
                "HFO Annotation Selected",
                "Selected an Artifact review label before saving the annotation.",
            )

            snapshot_path = self.export_dir / "hfo_review_snapshot.png"
            with _patched_save_dialog(snapshot_path):
                annotation_window.export_snapshot()
            annotation_window.update_button_clicked()
            _process_events(self.app)
            active_run = window.model.backend.analysis_session.get_active_run()
            assert active_run is not None
            assert int(active_run.event_features.annotated[0]) == 1
            assert int(active_run.event_features.artifact_annotations[0]) == 1
            assert snapshot_path.exists()
            self.capture(
                annotation_window,
                "20_hfo_annotation_saved",
                "HFO Annotation Saved",
                "Saved the annotation, advanced to the next event, and exported a review snapshot.",
                {"snapshot_export": str(snapshot_path)},
            )

            session_path = self.export_dir / "hfo_session.pybrain"
            window.model._save_to_npz(session_path, None)
            assert session_path.exists()

            restored_window = self.create_main_window()
            try:
                restored_window.model._offer_direct_annotation = lambda: None
                restored_window.model._load_from_npz(session_path, None)
                restored_window.model.load_from_npz_finished()
                restored_active = restored_window.model.backend.analysis_session.get_active_run()
                assert restored_active is not None
                self.capture(
                    restored_window,
                    "21_hfo_session_restored",
                    "HFO Session Restored",
                    "Restored the exported HFO session and verified that runs, exports, and review state come back.",
                    {
                        "session_export": str(session_path),
                        "accepted_detector": restored_window.model.backend.get_decision_summary()["accepted_detector"],
                        "annotated_first_event": int(restored_active.event_features.annotated[0]),
                    },
                )
            finally:
                _close_widget(restored_window, self.app)

            return {
                "excel_export": excel_path,
                "report_export": report_path,
                "snapshot_export": snapshot_path,
                "session_export": session_path,
            }
        finally:
            _close_widget(annotation_window, self.app)
            _close_widget(run_stats_dialog, self.app)
            _close_widget(window, self.app)

    def run_spindle_and_spike_flow(self, spindle_input_path):
        if not has_yasa():
            raise RuntimeError("Spindle walkthrough requires the optional 'yasa' dependency.")

        window = self.create_main_window()
        annotation_window = None
        try:
            window.combo_box_biomarker.setCurrentText("Spindle")
            _process_events(self.app)
            self.capture(
                window,
                "30_spindle_home",
                "Spindle Workspace Home",
                "Switched from HFO into the dedicated Spindle workspace.",
            )

            _load_recording_into_model(window.model, spindle_input_path)
            self.capture(
                window,
                "31_spindle_loaded",
                "Spindle Recording Loaded",
                "Loaded a synthetic spindle recording with visible spindle bursts.",
                {"recording": str(spindle_input_path)},
            )

            _run_filter_via_model(window.model)
            self.capture(
                window,
                "32_spindle_filtered",
                "Spindle Filtered",
                "Filtered the spindle recording in the main workspace.",
            )

            _run_spindle_detector_via_model(window.model)
            assert window.model.backend.event_features.get_num_biomarker() >= 1
            window.model.set_classifier_param_cpu_default()
            _run_classifier_via_model(window.model, use_spike=True, use_ehfo=False)
            self.capture(
                window,
                "33_spindle_classified",
                "Spindle Classified",
                "Ran YASA detection plus artifact and spike-associated spindle classification.",
                {
                    "num_events": int(window.model.backend.event_features.get_num_biomarker()),
                    "artifact_predicted": bool(window.model.backend.event_features.artifact_predicted),
                    "spike_predicted": bool(window.model.backend.event_features.spike_predicted),
                },
            )

            spindle_excel_path = self.export_dir / "spindle_clinical_summary.xlsx"
            with _patched_save_dialog(spindle_excel_path):
                window.model.save_to_excel()
            assert spindle_excel_path.exists()

            window.model.open_annotation()
            _process_events(self.app)
            annotation_window = _find_annotation_window(window)
            _resize_widget(annotation_window, ANNOTATION_CAPTURE_SIZE, self.app)
            self.capture(
                annotation_window,
                "34_spindle_annotation_open",
                "Spindle Annotation Open",
                "Opened the spindle review window for the active YASA run.",
            )

            annotation_window.select_annotation_option("Spike")
            self.capture(
                annotation_window,
                "35_spindle_annotation_selected",
                "Spindle Annotation Selected",
                "Selected a Spike review label in the spindle annotation window.",
                {"excel_export": str(spindle_excel_path)},
            )

            annotation_window.update_button_clicked()
            _process_events(self.app)
            active_run = window.model.backend.analysis_session.get_active_run()
            assert active_run is not None
            assert int(active_run.event_features.annotated[0]) == 1
            assert int(active_run.event_features.spike_annotations[0]) == 1
            self.capture(
                annotation_window,
                "36_spindle_annotation_saved",
                "Spindle Annotation Saved",
                "Saved the spindle annotation and advanced to the next review item.",
            )

            window.combo_box_biomarker.setCurrentText("Spike")
            _process_events(self.app)
            assert window.model.biomarker_type == "Spike"
            assert window.detector_run_button.isEnabled() is False
            assert window.classifier_run_button.isEnabled() is False
            self.capture(
                window,
                "40_spike_review_only",
                "Spike Review-Only Branch",
                "Switched into the Spike branch to confirm the review-only workflow and disabled automation controls.",
                {
                    "detector_button_text": window.detector_run_button.text(),
                    "classifier_button_text": window.classifier_run_button.text(),
                },
            )
        finally:
            _close_widget(annotation_window, self.app)
            _close_widget(window, self.app)

    def write_report(self, quick_outputs, hfo_outputs):
        report_path = self.output_root / "report.md"
        payload_path = self.output_root / "summary.json"
        relative = lambda path: path.relative_to(self.output_root).as_posix()

        lines = [
            "# UI Walkthrough Report",
            "",
            f"- Generated: `{datetime.now().isoformat()}`",
            f"- Root: `{self.output_root}`",
            "",
            "## Exported Files",
            "",
            f"- Quick detection Excel: `{quick_outputs['excel_output']}`",
            f"- Quick detection session: `{quick_outputs['npz_output']}`",
            f"- HFO clinical summary: `{hfo_outputs['excel_export']}`",
            f"- HFO HTML report: `{hfo_outputs['report_export']}`",
            f"- HFO review snapshot: `{hfo_outputs['snapshot_export']}`",
            f"- HFO session export: `{hfo_outputs['session_export']}`",
            "",
            "## Steps",
            "",
        ]

        for index, step in enumerate(self.steps, start=1):
            lines.append(f"### {index}. {step['title']}")
            lines.append("")
            lines.append(f"![{step['title']}]({relative(step['screenshot'])})")
            lines.append("")
            if step["note"]:
                lines.append(step["note"])
                lines.append("")
            if step["extra"]:
                for key, value in step["extra"].items():
                    lines.append(f"- `{key}`: `{value}`")
                lines.append("")

        report_path.write_text("\n".join(lines), encoding="utf-8")

        payload = {
            "generated_at": datetime.now().isoformat(),
            "output_root": str(self.output_root),
            "steps": [
                {
                    "name": step["name"],
                    "title": step["title"],
                    "note": step["note"],
                    "screenshot": str(step["screenshot"]),
                    "extra": step["extra"],
                }
                for step in self.steps
            ],
        }
        payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return report_path, payload_path

    def run(self):
        hfo_input_path = _build_hfo_midwindow_recording(self.input_dir)
        spindle_input_path = _build_spindle_recording(self.input_dir)
        quick_outputs = self.run_quick_detection_flow(hfo_input_path)
        hfo_outputs = self.run_hfo_main_flow(hfo_input_path)
        self.run_spindle_and_spike_flow(spindle_input_path)
        report_path, payload_path = self.write_report(quick_outputs, hfo_outputs)
        print(f"[report] {report_path}")
        print(f"[summary] {payload_path}")
        print(f"[done] Walkthrough artifacts saved under {self.output_root}")


def main():
    runner = UIWalkthroughCapture()
    runner.run()


if __name__ == "__main__":
    main()
