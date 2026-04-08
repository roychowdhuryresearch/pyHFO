from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np

from src.hfo_feature import HFO_Feature
from src.param.param_classifier import ParamClassifier
from src.param.param_detector import ParamDetector
from src.param.param_filter import ParamFilter, ParamFilterSpindle
from src.spike_feature import SpikeFeature
from src.spindle_feature import SpindleFeature


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _restore_filter_param(biomarker_type: str, payload: dict | None):
    if not payload:
        return None
    if biomarker_type == "Spindle":
        return ParamFilterSpindle.from_dict(payload)
    return ParamFilter.from_dict(payload)


def _restore_detector_param(payload: dict | None):
    if not payload:
        return None
    return ParamDetector.from_dict(payload)


def _restore_classifier_param(payload: dict | None):
    if not payload:
        return None
    return ParamClassifier.from_dict(payload)


def _restore_event_features(biomarker_type: str, payload: dict | None):
    if not payload:
        return None
    if biomarker_type == "Spike":
        feature = SpikeFeature.from_dict(payload)
        feature.artifact_predictions = np.array(payload.get("artifact_predictions", getattr(feature, "artifact_predictions", np.array([]))))
        feature.accepted_predictions = np.array(payload.get("accepted_predictions", getattr(feature, "accepted_predictions", np.array([]))))
        feature.artifact_annotations = np.array(payload.get("artifact_annotations", getattr(feature, "artifact_annotations", np.array([]))))
        feature.accepted_annotations = np.array(payload.get("accepted_annotations", getattr(feature, "accepted_annotations", np.array([]))))
        feature.annotated = np.array(payload.get("annotated", getattr(feature, "annotated", np.array([]))))
        feature._refresh_annotation_counts()
        return feature
    if biomarker_type == "Spindle":
        feature = SpindleFeature.from_dict(payload)
        artifact_predictions = np.array(payload.get("artifact_predictions", getattr(feature, "artifact_predictions", np.array([]))))
        spike_predictions = np.array(payload.get("spike_predictions", getattr(feature, "spike_predictions", np.array([]))))
        feature.update_pred(artifact_predictions, spike_predictions)
        feature.artifact_annotations = np.array(payload.get("artifact_annotations", getattr(feature, "artifact_annotations", np.array([]))))
        feature.spike_annotations = np.array(payload.get("spike_annotations", getattr(feature, "spike_annotations", np.array([]))))
        feature.annotated = np.array(payload.get("annotated", getattr(feature, "annotated", np.array([]))))
        return feature

    feature = HFO_Feature.from_dict(payload)
    artifact_predictions = np.array(payload.get("artifact_predictions", getattr(feature, "artifact_predictions", np.array([]))))
    spike_predictions = np.array(payload.get("spike_predictions", getattr(feature, "spike_predictions", np.array([]))))
    ehfo_predictions = np.array(payload.get("ehfo_predictions", getattr(feature, "ehfo_predictions", np.array([]))))
    feature.update_pred(artifact_predictions, spike_predictions, ehfo_predictions)
    feature.artifact_annotations = np.array(payload.get("artifact_annotations", getattr(feature, "artifact_annotations", np.array([]))))
    feature.pathological_annotations = np.array(payload.get("pathological_annotations", getattr(feature, "pathological_annotations", np.array([]))))
    feature.physiological_annotations = np.array(payload.get("physiological_annotations", getattr(feature, "physiological_annotations", np.array([]))))
    feature.annotated = np.array(payload.get("annotated", getattr(feature, "annotated", np.array([]))))
    return feature


def serialize_event_features(biomarker_type: str, event_features):
    if event_features is None:
        return None
    payload = event_features.to_dict()
    payload["annotated"] = np.array(getattr(event_features, "annotated", np.array([])))
    if biomarker_type == "Spike":
        payload["artifact_annotations"] = np.array(getattr(event_features, "artifact_annotations", np.array([])))
        payload["accepted_annotations"] = np.array(getattr(event_features, "accepted_annotations", np.array([])))
        payload["artifact_predictions"] = np.array(getattr(event_features, "artifact_predictions", np.array([])))
        payload["accepted_predictions"] = np.array(getattr(event_features, "accepted_predictions", np.array([])))
        return payload
    if biomarker_type == "Spindle":
        payload["artifact_annotations"] = np.array(getattr(event_features, "artifact_annotations", np.array([])))
        payload["spike_annotations"] = np.array(getattr(event_features, "spike_annotations", np.array([])))
    else:
        payload["artifact_annotations"] = np.array(getattr(event_features, "artifact_annotations", np.array([])))
        payload["pathological_annotations"] = np.array(getattr(event_features, "pathological_annotations", np.array([])))
        payload["physiological_annotations"] = np.array(getattr(event_features, "physiological_annotations", np.array([])))
    return payload


def _feature_array(event_features, attr_name):
    if event_features is None:
        return np.array([])
    if hasattr(event_features, "get_visible_array"):
        try:
            return np.array(event_features.get_visible_array(attr_name))
        except Exception:
            pass
    return np.array(getattr(event_features, attr_name, np.array([])))


def _feature_event_tuples(event_features):
    if event_features is None:
        return set()
    if hasattr(event_features, "iter_visible_events"):
        try:
            return {
                (str(channel_name), int(start), int(end))
                for channel_name, start, end in event_features.iter_visible_events()
            }
        except Exception:
            pass
    channel_names = np.array(getattr(event_features, "channel_names", np.array([])))
    starts = np.array(getattr(event_features, "starts", np.array([])))
    ends = np.array(getattr(event_features, "ends", np.array([])))
    return {
        (str(channel_name), int(start), int(end))
        for channel_name, start, end in zip(channel_names, starts, ends)
    }


def _feature_event_rows(event_features):
    if event_features is None:
        return []
    channel_names = _feature_array(event_features, "channel_names")
    starts = _feature_array(event_features, "starts")
    ends = _feature_array(event_features, "ends")
    if len(channel_names) == 0 or len(starts) == 0 or len(ends) == 0:
        return []

    try:
        sample_freq = float(getattr(event_features, "sample_freq", 0) or 0)
    except (TypeError, ValueError):
        sample_freq = 0.0
    if sample_freq <= 0:
        sample_freq = 1.0

    rows = []
    for channel_name, start, end in zip(channel_names, starts, ends):
        start_index = int(start)
        end_index = int(end)
        if end_index <= start_index:
            end_index = start_index + 1
        rows.append(
            {
                "channel_name": str(channel_name),
                "start_index": start_index,
                "end_index": end_index,
                "start_seconds": float(start_index) / sample_freq,
                "end_seconds": float(end_index) / sample_freq,
            }
        )
    return rows


def _events_temporally_overlap(left_event: dict[str, Any], right_event: dict[str, Any]) -> bool:
    return (
        min(left_event["end_seconds"], right_event["end_seconds"])
        > max(left_event["start_seconds"], right_event["start_seconds"])
    )


def _count_temporal_overlap_matches(left_rows, right_rows):
    if not left_rows or not right_rows:
        return 0, 0

    left_by_channel = defaultdict(list)
    right_by_channel = defaultdict(list)
    for row in left_rows:
        left_by_channel[row["channel_name"]].append(row)
    for row in right_rows:
        right_by_channel[row["channel_name"]].append(row)

    shared_channels = 0
    match_count = 0
    for channel_name in sorted(set(left_by_channel) & set(right_by_channel)):
        shared_channels += 1
        left_channel_rows = sorted(
            left_by_channel[channel_name],
            key=lambda row: (row["end_seconds"], row["start_seconds"]),
        )
        right_channel_rows = sorted(
            enumerate(right_by_channel[channel_name]),
            key=lambda item: (item[1]["end_seconds"], item[1]["start_seconds"]),
        )
        used_right_indexes = set()
        for left_event in left_channel_rows:
            for right_index, right_event in right_channel_rows:
                if right_index in used_right_indexes:
                    continue
                if right_event["end_seconds"] <= left_event["start_seconds"]:
                    continue
                if not _events_temporally_overlap(left_event, right_event):
                    continue
                used_right_indexes.add(right_index)
                match_count += 1
                break
    return match_count, shared_channels


def summarize_event_features(event_features):
    if event_features is None:
        return {"num_events": 0, "num_channels": 0, "top_channels": []}
    channel_names = _feature_array(event_features, "channel_names")
    starts = _feature_array(event_features, "starts")
    counter = Counter(channel_names.tolist())
    top_channels = [
        {"channel_name": channel_name, "count": count}
        for channel_name, count in counter.most_common(10)
    ]
    summary = {
        "num_events": int(len(getattr(event_features, "starts", []))),
        "num_channels": int(len(counter)),
        "top_channels": top_channels,
    }
    summary["num_events"] = int(len(starts))
    if hasattr(event_features, "get_overlap_review_summary"):
        try:
            summary.update(event_features.get_overlap_review_summary())
        except Exception:
            pass
    return summary


def build_run_comparison(selected_runs: list["DetectionRun"]):
    if len(selected_runs) < 2:
        return {"runs": [], "pairwise_overlap": []}

    run_summaries = []
    event_rows_by_run = {}
    for run in selected_runs:
        run.refresh_summary()
        event_rows_by_run[run.run_id] = _feature_event_rows(run.event_features)
        run_summaries.append(
            {
                "run_id": run.run_id,
                "display_name": run.display_name,
                "detector_name": run.detector_name,
                "biomarker_type": run.biomarker_type,
                "num_events": run.summary.get("num_events", 0),
                "num_channels": run.summary.get("num_channels", 0),
            }
        )

    pairwise = []
    for i, left in enumerate(selected_runs):
        for right in selected_runs[i + 1:]:
            left_events = event_rows_by_run[left.run_id]
            right_events = event_rows_by_run[right.run_id]
            overlap_count, shared_channels = _count_temporal_overlap_matches(left_events, right_events)
            union_count = max(0, len(left_events) + len(right_events) - overlap_count)
            pairwise.append(
                {
                    "left_run_id": left.run_id,
                    "right_run_id": right.run_id,
                    "left_detector": left.detector_name,
                    "right_detector": right.detector_name,
                    "left_biomarker": left.biomarker_type,
                    "right_biomarker": right.biomarker_type,
                    "left_label": f"{left.biomarker_type} • {left.detector_name}",
                    "right_label": f"{right.biomarker_type} • {right.detector_name}",
                    "comparison_mode": "temporal_overlap",
                    "shared_channels": shared_channels,
                    "overlap_events": overlap_count,
                    "union_events": union_count,
                    "jaccard": (overlap_count / union_count) if union_count else 1.0,
                    "left_only": max(0, len(left_events) - overlap_count),
                    "right_only": max(0, len(right_events) - overlap_count),
                }
            )
    return {"runs": run_summaries, "pairwise_overlap": pairwise}


@dataclass
class DetectionRun:
    run_id: str
    biomarker_type: str
    detector_name: str
    display_name: str
    created_at: str
    selected_channels: list[str] = field(default_factory=list)
    param_filter: Any = None
    param_detector: Any = None
    param_classifier: Any = None
    event_features: Any = None
    detector_output: Any = None
    classified: bool = False
    summary: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        biomarker_type: str,
        detector_name: str,
        selected_channels: Any,
        param_filter: Any = None,
        param_detector: Any = None,
        param_classifier: Any = None,
        event_features: Any = None,
        detector_output: Any = None,
        classified: bool = False,
    ) -> "DetectionRun":
        stamp = _now_iso()
        detector_label = detector_name or biomarker_type
        display_name = f"{detector_label} {stamp.replace('T', ' ').replace('Z', ' UTC')}"
        channels = list(np.array(selected_channels).tolist()) if selected_channels is not None else []
        return cls(
            run_id=str(uuid4()),
            biomarker_type=biomarker_type,
            detector_name=detector_name,
            display_name=display_name,
            created_at=stamp,
            selected_channels=channels,
            param_filter=param_filter,
            param_detector=param_detector,
            param_classifier=param_classifier,
            event_features=event_features,
            detector_output=detector_output,
            classified=classified,
            summary=summarize_event_features(event_features),
        )

    def refresh_summary(self):
        self.summary = summarize_event_features(self.event_features)

    def to_dict(self):
        self.refresh_summary()
        return {
            "run_id": self.run_id,
            "biomarker_type": self.biomarker_type,
            "detector_name": self.detector_name,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "selected_channels": list(self.selected_channels),
            "param_filter": self.param_filter.to_dict() if self.param_filter else None,
            "param_detector": self.param_detector.to_dict() if self.param_detector else None,
            "param_classifier": self.param_classifier.to_dict() if self.param_classifier else None,
            "event_features": serialize_event_features(self.biomarker_type, self.event_features),
            "detector_output": self.detector_output,
            "classified": self.classified,
            "summary": dict(self.summary),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DetectionRun":
        biomarker_type = payload["biomarker_type"]
        run = cls(
            run_id=payload["run_id"],
            biomarker_type=biomarker_type,
            detector_name=payload.get("detector_name", biomarker_type),
            display_name=payload.get("display_name", payload.get("detector_name", biomarker_type)),
            created_at=payload.get("created_at", _now_iso()),
            selected_channels=list(payload.get("selected_channels", [])),
            param_filter=_restore_filter_param(biomarker_type, payload.get("param_filter")),
            param_detector=_restore_detector_param(payload.get("param_detector")),
            param_classifier=_restore_classifier_param(payload.get("param_classifier")),
            event_features=_restore_event_features(biomarker_type, payload.get("event_features")),
            detector_output=payload.get("detector_output"),
            classified=payload.get("classified", False),
            summary=dict(payload.get("summary", {})),
        )
        run.refresh_summary()
        return run


@dataclass
class AnalysisSession:
    biomarker_type: str
    active_run_id: str | None = None
    accepted_run_id: str | None = None
    visible_run_ids: list[str] = field(default_factory=list)
    runs: dict[str, DetectionRun] = field(default_factory=dict)

    def add_run(self, run: DetectionRun):
        self.runs[run.run_id] = run
        self.active_run_id = run.run_id
        if run.run_id not in self.visible_run_ids:
            self.visible_run_ids.append(run.run_id)

    def get_active_run(self) -> DetectionRun | None:
        if self.active_run_id is None:
            return None
        return self.runs.get(self.active_run_id)

    def activate_run(self, run_id: str):
        if run_id not in self.runs:
            raise KeyError(f"Unknown run id: {run_id}")
        self.active_run_id = run_id
        # Activating a run should always make its overlay visible in review mode.
        if run_id not in self.visible_run_ids:
            self.visible_run_ids.append(run_id)

    def accept_run(self, run_id: str):
        if run_id not in self.runs:
            raise KeyError(f"Unknown run id: {run_id}")
        self.accepted_run_id = run_id

    def get_accepted_run(self) -> DetectionRun | None:
        if self.accepted_run_id is None:
            return None
        return self.runs.get(self.accepted_run_id)

    def get_run(self, run_id: str) -> DetectionRun | None:
        return self.runs.get(run_id)

    def list_runs(self):
        return list(self.runs.values())

    def is_run_visible(self, run_id: str) -> bool:
        return run_id in self.visible_run_ids

    def set_run_visible(self, run_id: str, visible: bool):
        if run_id not in self.runs:
            raise KeyError(f"Unknown run id: {run_id}")
        if visible:
            if run_id not in self.visible_run_ids:
                self.visible_run_ids.append(run_id)
        else:
            self.visible_run_ids = [visible_id for visible_id in self.visible_run_ids if visible_id != run_id]

    def set_visible_runs(self, run_ids: list[str]):
        valid_ids = [run_id for run_id in run_ids if run_id in self.runs]
        self.visible_run_ids = valid_ids

    def get_visible_runs(self):
        visible = [self.runs[run_id] for run_id in self.visible_run_ids if run_id in self.runs]
        if visible:
            return visible
        active_run = self.get_active_run()
        return [active_run] if active_run is not None else []

    def get_channel_ranking(self, run_id: str | None = None):
        run = self.get_run(run_id) if run_id else self.get_active_run()
        if run is None or run.event_features is None:
            return []
        event_features = run.event_features
        channel_names = _feature_array(event_features, "channel_names")
        if len(channel_names) == 0:
            return []

        rows = []
        unique_channels = np.unique(channel_names)
        artifact_predictions = _feature_array(event_features, "artifact_predictions")
        spike_predictions = _feature_array(event_features, "spike_predictions")
        ehfo_predictions = _feature_array(event_features, "ehfo_predictions") if hasattr(event_features, "ehfo_predictions") else np.array([])
        annotated = _feature_array(event_features, "annotated")
        artifact_annotations = _feature_array(event_features, "artifact_annotations")

        for channel_name in unique_channels:
            idx = np.where(channel_names == channel_name)[0]
            row = {
                "channel_name": channel_name,
                "total_events": int(len(idx)),
                "artifact_predicted": int(np.sum(artifact_predictions[idx] < 1)) if len(artifact_predictions) else 0,
                "accepted_predicted": int(np.sum(artifact_predictions[idx] > 0)) if len(artifact_predictions) else int(len(idx)),
                "spike_associated": int(np.sum(spike_predictions[idx] == 1)) if len(spike_predictions) else 0,
                "annotated_events": int(np.sum(annotated[idx] > 0)) if len(annotated) else 0,
                "artifact_annotated": int(np.sum(artifact_annotations[idx] > 0)) if len(artifact_annotations) else 0,
            }
            if len(ehfo_predictions):
                row["ehfo_predicted"] = int(np.sum(ehfo_predictions[idx] == 1))
            rows.append(row)

        rows.sort(key=lambda row: (-row["accepted_predicted"], -row["total_events"], row["channel_name"]))
        return rows

    def compare_runs(self, run_ids: list[str] | None = None):
        selected_runs = [self.runs[run_id] for run_id in (run_ids or list(self.runs.keys())) if run_id in self.runs]
        return build_run_comparison(selected_runs)

    def to_dict(self):
        return {
            "biomarker_type": self.biomarker_type,
            "active_run_id": self.active_run_id,
            "accepted_run_id": self.accepted_run_id,
            "visible_run_ids": list(self.visible_run_ids),
            "runs": [run.to_dict() for run in self.runs.values()],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnalysisSession":
        session = cls(
            biomarker_type=payload["biomarker_type"],
            active_run_id=payload.get("active_run_id"),
            accepted_run_id=payload.get("accepted_run_id"),
            visible_run_ids=list(payload.get("visible_run_ids", [])),
        )
        for run_payload in payload.get("runs", []):
            run = DetectionRun.from_dict(run_payload)
            session.runs[run.run_id] = run
        if session.active_run_id not in session.runs and session.runs:
            session.active_run_id = next(iter(session.runs))
        if not session.visible_run_ids and session.active_run_id is not None:
            session.visible_run_ids = [session.active_run_id]
        else:
            session.visible_run_ids = [run_id for run_id in session.visible_run_ids if run_id in session.runs]
        return session
