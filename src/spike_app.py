from __future__ import annotations

import numpy as np

from src.param.param_classifier import ParamClassifier
from src.param.param_detector import ParamDetector
from src.param.param_filter import ParamFilter
from src.spike_feature import SpikeFeature
from src.spindle_app import SpindleApp
from src.utils.analysis_session import AnalysisSession
from src.utils.app_state import build_base_checkpoint, checkpoint_array, checkpoint_get
from src.utils.session_store import load_session_checkpoint, save_session_checkpoint
from src.utils.utils_detector import set_spike_rms_ll_detector
from src.utils.utils_montage import source_channel_names


class SpikeApp(SpindleApp):
    def __init__(self):
        super().__init__()
        self.biomarker_type = "Spike"
        self.param_filter = None
        self.classifier = None
        self.classified = False
        self.Spikes = None
        self.analysis_session = AnalysisSession(self.biomarker_type)

    def set_detector(self, param: ParamDetector):
        self.param_detector = param
        self.param_detector.detector_param.sample_freq = self.sample_freq
        if self.param_filter is not None:
            self.param_detector.detector_param.pass_band = float(self.param_filter.fp)
            self.param_detector.detector_param.stop_band = float(self.param_filter.fs)
        if param.detector_type.lower() == "rms/ll":
            self.detector = set_spike_rms_ll_detector(param.detector_param)
        else:
            raise ValueError(f"Unsupported spike detector: {param.detector_type}")

    def detect_biomarker(self, param_filter: ParamFilter = None, param_detector: ParamDetector = None):
        if param_filter is not None and not self.has_filtered_data():
            self.filter_eeg_data(param_filter)
        if param_detector is not None:
            self.set_detector(param_detector)
        if self.filter_data is None or len(self.filter_data) == 0:
            self.filter_eeg_data()
        self.event_channel_names, self.Spikes = self.detector.detect_multi_channels(
            self.filter_data,
            self.channel_names,
            filtered=True,
        )
        self.Spindles = self.Spikes
        self.event_features = SpikeFeature.construct(
            self.event_channel_names,
            self.Spikes,
            self.param_detector.detector_type,
            self.sample_freq,
            freq_range=[self.param_filter.fp if self.param_filter else 1, self.param_filter.fs if self.param_filter else 80],
        )
        self.detected = True
        self.classified = False
        self.capture_current_run()

    def export_app(self, path):
        self.sync_active_run()
        checkpoint = build_base_checkpoint(self, self.biomarker_type)
        checkpoint["analysis_session"] = self.analysis_session.to_dict()
        checkpoint.update(
            {
                "Spikes": self.Spikes,
                "param_detector": self.param_detector.to_dict() if self.param_detector else None,
                "Spike_features": self.event_features.to_dict() if self.event_features else None,
                "param_classifier": self.param_classifier.to_dict() if self.param_classifier else None,
                "artifact_predictions": np.array(self.event_features.artifact_predictions) if self.event_features else np.array([]),
                "accepted_predictions": np.array(self.event_features.accepted_predictions) if self.event_features else np.array([]),
                "artifact_annotations": np.array(self.event_features.artifact_annotations) if self.event_features else np.array([]),
                "accepted_annotations": np.array(self.event_features.accepted_annotations) if self.event_features else np.array([]),
                "annotated": np.array(self.event_features.annotated) if self.event_features else np.array([]),
            }
        )
        save_session_checkpoint(path, checkpoint)

    @staticmethod
    def import_app(path):
        checkpoint = load_session_checkpoint(path)
        app = SpikeApp()
        app.load_checkpoint(checkpoint)
        return app

    def activate_run(self, run_id):
        self.analysis_session.activate_run(run_id)
        run = self.analysis_session.get_active_run()
        if run is None:
            return
        self.param_filter = run.param_filter
        self.param_detector = run.param_detector
        self.param_classifier = run.param_classifier
        self.event_features = run.event_features
        self.Spikes = run.detector_output
        self.Spindles = self.Spikes
        self.detected = self.event_features is not None
        self.classified = run.classified

    def load_checkpoint(self, checkpoint):
        self.n_jobs = checkpoint_get(checkpoint, "n_jobs", self.n_jobs)
        self.eeg_data = checkpoint_array(checkpoint, "eeg_data")
        self.eeg_data_un60 = checkpoint_array(checkpoint, "eeg_data_un60", self.eeg_data)
        self.eeg_data_60 = checkpoint_array(checkpoint, "eeg_data_60", None)
        self.edf_param = checkpoint_get(checkpoint, "edf_param")
        self.sample_freq = checkpoint_get(checkpoint, "sample_freq", 0)
        self.channel_names = checkpoint_array(checkpoint, "channel_names")
        self.recording_channel_names = checkpoint_array(
            checkpoint,
            "recording_channel_names",
            np.array(source_channel_names(self.channel_names)),
        )
        self.classified = checkpoint_get(checkpoint, "classified", False)
        self.filtered = checkpoint_get(checkpoint, "filtered", False)
        self.detected = checkpoint_get(checkpoint, "detected", False)

        if self.filtered:
            filter_dict = checkpoint_get(checkpoint, "param_filter")
            if filter_dict:
                self.param_filter = ParamFilter.from_dict(filter_dict)
            self.filter_data = checkpoint_array(checkpoint, "filter_data")
            self.filter_data_un60 = checkpoint_array(checkpoint, "filter_data_un60", self.filter_data)
            self.filter_data_60 = checkpoint_array(checkpoint, "filter_data_60", None)

        if self.classified:
            classifier_dict = checkpoint_get(checkpoint, "param_classifier")
            if classifier_dict:
                self.param_classifier = ParamClassifier.from_dict(classifier_dict)

        session_payload = checkpoint_get(checkpoint, "analysis_session")
        if session_payload:
            self.analysis_session = AnalysisSession.from_dict(session_payload)
            if self.analysis_session.get_active_run() is not None:
                self.activate_run(self.analysis_session.active_run_id)
                return

        self.analysis_session = AnalysisSession(self.biomarker_type)
        if self.detected:
            self.Spikes = checkpoint_array(checkpoint, "Spikes")
            self.Spindles = self.Spikes
            detector_dict = checkpoint_get(checkpoint, "param_detector")
            if detector_dict:
                self.param_detector = ParamDetector.from_dict(detector_dict)
            feature_dict = checkpoint_get(checkpoint, "Spike_features")
            if feature_dict:
                self.event_features = SpikeFeature.from_dict(feature_dict)
                self.capture_current_run()
