import numpy as np
import pandas as pd


class HFO_Feature:
    OVERLAP_ACTION_DISABLED = "disabled"
    OVERLAP_ACTION_TAG = "tag_only"
    OVERLAP_ACTION_HIDE = "hide_tagged"
    DEFAULT_OVERLAP_TAG = "Cross-channel overlap"

    def __init__(
        self,
        channel_names,
        interval,
        features=[],
        HFO_type="STE",
        sample_freq=2000,
        freq_range=[10, 500],
        time_range=[0, 1000],
        feature_size=224,
        raw_spectrums=None,
    ):
        self.channel_names = np.array(channel_names)
        self.features = features
        self.spike_predictions = np.array([])
        self.ehfo_predictions = np.array([])

        interval = np.array(interval)
        if interval.size == 0:
            self.starts = np.array([], dtype=int)
            self.ends = np.array([], dtype=int)
            self.artifact_predictions = np.array([])
            self.artifact_annotations = np.array([])
            self.pathological_annotations = np.array([])
            self.physiological_annotations = np.array([])
            self.annotated = np.array([])
        else:
            self.starts = np.array(interval[:, 0]).astype(int)
            self.ends = np.array(interval[:, 1]).astype(int)
            self.artifact_predictions = np.zeros(self.starts.shape)
            self.artifact_annotations = np.zeros(self.starts.shape)
            self.pathological_annotations = np.zeros(self.starts.shape)
            self.physiological_annotations = np.zeros(self.starts.shape)
            self.annotated = np.zeros(self.starts.shape)

        self.HFO_type = HFO_type
        self.sample_freq = sample_freq
        self.feature_size = feature_size
        self.freq_range = freq_range
        self.time_range = time_range
        self.num_artifact = 0
        self.num_spike = 0
        self.num_ehfo = 0
        self.num_HFO = len(self.starts)
        self.num_real = 0
        self.index = 0
        self.artifact_predicted = False
        self.spike_predicted = False
        self.ehfo_predicted = False
        self.raw_spectrums = raw_spectrums
        self._channel_index_cache = None
        self.view_state_token = 0

        self.overlap_review_settings = self._normalize_overlap_review_settings()
        self.overlap_tagged = np.zeros(self.starts.shape, dtype=int)
        self.overlap_group_ids = np.full(self.starts.shape, -1, dtype=int)
        self.overlap_channel_counts = np.zeros(self.starts.shape, dtype=int)
        self.overlap_event_counts = np.zeros(self.starts.shape, dtype=int)
        self.overlap_visibility_mask = np.ones(self.starts.shape, dtype=bool)
        self._repair_overlap_view_state()

    def __str__(self):
        return "HFO_Feature: {} HFOs, {} artifacts, {} spkHFOs, {} eHFOs, {} real HFOs".format(
            self.num_HFO,
            self.get_num_artifact(),
            self.get_num_spike(),
            self.get_num_ehfo(),
            self.get_num_real(),
        )

    @staticmethod
    def construct(
        channel_names,
        start_end,
        HFO_type="STE",
        sample_freq=2000,
        freq_range=[10, 500],
        time_range=[0, 1000],
        feature_size=224,
    ):
        """
        Construct HFO_Feature object from detector output
        """
        channel_names = np.concatenate([[channel_names[i]] * len(start_end[i]) for i in range(len(channel_names))])
        start_end = [se for se in start_end if len(se) > 0]
        start_end = np.concatenate(start_end) if start_end else np.empty((0, 2), dtype=int)

        valid_indices = np.where((start_end[:, 1] - start_end[:, 0]) < sample_freq)[0] if start_end.size > 0 else np.array([])
        start_end = start_end[valid_indices] if valid_indices.size > 0 else np.empty((0, 2), dtype=int)
        channel_names = channel_names[valid_indices] if valid_indices.size > 0 else np.array([])
        return HFO_Feature(channel_names, start_end, np.array([]), HFO_type, sample_freq, freq_range, time_range, feature_size)

    @classmethod
    def default_overlap_review_settings(cls):
        return {
            "action": cls.OVERLAP_ACTION_DISABLED,
            "min_overlap_ms": 0.0,
            "min_channels": 2,
            "tag_name": cls.DEFAULT_OVERLAP_TAG,
        }

    def _normalize_overlap_review_settings(self, settings=None):
        normalized = dict(self.default_overlap_review_settings())
        if settings:
            normalized.update(
                {
                    key: value
                    for key, value in dict(settings).items()
                    if key in {"action", "min_overlap_ms", "min_channels", "tag_name"}
                }
            )

        action = str(normalized.get("action", self.OVERLAP_ACTION_DISABLED) or self.OVERLAP_ACTION_DISABLED).strip().lower()
        if action not in {
            self.OVERLAP_ACTION_DISABLED,
            self.OVERLAP_ACTION_TAG,
            self.OVERLAP_ACTION_HIDE,
        }:
            action = self.OVERLAP_ACTION_DISABLED

        try:
            min_overlap_ms = float(normalized.get("min_overlap_ms", 0.0) or 0.0)
        except (TypeError, ValueError):
            min_overlap_ms = 0.0
        min_overlap_ms = max(0.0, min_overlap_ms)

        try:
            min_channels = int(normalized.get("min_channels", 2) or 2)
        except (TypeError, ValueError):
            min_channels = 2
        min_channels = max(2, min_channels)

        tag_name = str(normalized.get("tag_name", self.DEFAULT_OVERLAP_TAG) or self.DEFAULT_OVERLAP_TAG).strip()
        if not tag_name:
            tag_name = self.DEFAULT_OVERLAP_TAG

        return {
            "action": action,
            "min_overlap_ms": min_overlap_ms,
            "min_channels": min_channels,
            "tag_name": tag_name,
        }

    def _repair_overlap_view_state(self):
        count = len(self.starts)
        self.overlap_review_settings = self._normalize_overlap_review_settings(self.overlap_review_settings)
        self.overlap_tagged = self._resize_state_array(self.overlap_tagged, count, fill_value=0, dtype=int)
        self.overlap_group_ids = self._resize_state_array(self.overlap_group_ids, count, fill_value=-1, dtype=int)
        self.overlap_channel_counts = self._resize_state_array(self.overlap_channel_counts, count, fill_value=0, dtype=int)
        self.overlap_event_counts = self._resize_state_array(self.overlap_event_counts, count, fill_value=0, dtype=int)
        default_visibility = np.ones(count, dtype=bool)
        if self.overlap_review_settings["action"] == self.OVERLAP_ACTION_HIDE and len(self.overlap_tagged) == count:
            default_visibility = np.array(self.overlap_tagged < 1, dtype=bool)
        self.overlap_visibility_mask = self._resize_state_array(
            self.overlap_visibility_mask,
            count,
            fill_value=True,
            dtype=bool,
            default_array=default_visibility,
        )
        if self.overlap_review_settings["action"] != self.OVERLAP_ACTION_HIDE:
            self.overlap_visibility_mask = np.ones(count, dtype=bool)
        self._ensure_current_visible_index()

    def _resize_state_array(self, values, count, fill_value=0, dtype=None, default_array=None):
        values = np.array(values)
        if len(values) == count:
            return values.astype(dtype) if dtype is not None else values
        if default_array is not None and len(default_array) == count:
            base = np.array(default_array)
        else:
            base = np.full(count, fill_value, dtype=dtype if dtype is not None else object)
        return base.astype(dtype) if dtype is not None else base

    def _increment_view_state_token(self):
        self.view_state_token += 1

    def get_raw_event_count(self):
        return int(len(self.starts))

    def get_visible_raw_indices(self):
        if len(self.starts) == 0:
            return np.array([], dtype=int)
        if len(self.overlap_visibility_mask) != len(self.starts):
            self._repair_overlap_view_state()
        return np.where(np.array(self.overlap_visibility_mask, dtype=bool))[0]

    def get_visible_event_count(self):
        return int(len(self.get_visible_raw_indices()))

    def get_num_biomarker(self):
        return self.get_visible_event_count()

    def get_visible_array(self, attr_name):
        values = np.array(getattr(self, attr_name, np.array([])))
        if len(values) == 0:
            return values
        if len(values) != len(self.starts):
            return values
        return values[self.get_visible_raw_indices()]

    def iter_visible_events(self):
        for raw_index in self.get_visible_raw_indices():
            yield (
                str(self.channel_names[raw_index]),
                int(self.starts[raw_index]),
                int(self.ends[raw_index]),
            )

    def get_overlap_review_settings(self):
        return dict(self.overlap_review_settings)

    def get_overlap_review_action_label(self):
        action = self.overlap_review_settings.get("action", self.OVERLAP_ACTION_DISABLED)
        if action == self.OVERLAP_ACTION_TAG:
            return "Tag later overlaps"
        if action == self.OVERLAP_ACTION_HIDE:
            return "Keep first, hide later overlaps"
        return "Disabled"

    def get_overlap_review_summary(self):
        tagged_count = int(np.sum(np.array(self.overlap_tagged) > 0)) if len(self.overlap_tagged) else 0
        hidden_count = int(np.sum(~np.array(self.overlap_visibility_mask, dtype=bool))) if len(self.overlap_visibility_mask) else 0
        group_ids = np.array(self.overlap_group_ids)
        valid_groups = group_ids[group_ids >= 0] if len(group_ids) else np.array([])
        group_count = int(len(np.unique(valid_groups))) if len(valid_groups) else 0
        return {
            "overlap_action": self.overlap_review_settings.get("action", self.OVERLAP_ACTION_DISABLED),
            "overlap_action_label": self.get_overlap_review_action_label(),
            "overlap_strategy": "keep_first",
            "overlap_tag_name": self.overlap_review_settings.get("tag_name", self.DEFAULT_OVERLAP_TAG),
            "overlap_min_overlap_ms": float(self.overlap_review_settings.get("min_overlap_ms", 0.0) or 0.0),
            "overlap_min_channels": int(self.overlap_review_settings.get("min_channels", 2) or 2),
            "overlap_tagged": tagged_count,
            "overlap_kept": group_count,
            "overlap_hidden": hidden_count,
            "overlap_visible": self.get_visible_event_count(),
            "overlap_groups": group_count,
        }

    def _find_overlap_components(self, min_overlap_ms, min_channels):
        if len(self.starts) < 2:
            return []

        min_overlap_samples = max(0, int(round(float(min_overlap_ms) * float(self.sample_freq) / 1000.0)))
        order = np.argsort(self.starts, kind="mergesort")
        parent = np.arange(len(self.starts), dtype=int)

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left, right):
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        starts = np.array(self.starts)
        ends = np.array(self.ends)
        channels = np.array(self.channel_names).astype(str)

        for order_position, left_index in enumerate(order):
            left_end = ends[left_index]
            right_position = order_position + 1
            while right_position < len(order):
                right_index = order[right_position]
                if starts[right_index] >= left_end:
                    break
                if channels[left_index] != channels[right_index]:
                    overlap_samples = min(ends[left_index], ends[right_index]) - max(starts[left_index], starts[right_index])
                    if overlap_samples > 0 and overlap_samples >= min_overlap_samples:
                        union(int(left_index), int(right_index))
                right_position += 1

        groups = {}
        for raw_index in range(len(self.starts)):
            root = find(raw_index)
            groups.setdefault(root, []).append(raw_index)

        qualified_groups = []
        for raw_indexes in groups.values():
            if len(raw_indexes) < 2:
                continue
            channel_count = len({str(channels[raw_index]) for raw_index in raw_indexes})
            if channel_count >= max(2, int(min_channels)):
                qualified_groups.append(sorted(raw_indexes))
        return qualified_groups

    def _select_group_primary_index(self, raw_indexes):
        ordered = sorted(
            [int(raw_index) for raw_index in raw_indexes],
            key=lambda raw_index: (
                int(self.starts[raw_index]),
                int(self.ends[raw_index]),
                int(raw_index),
            ),
        )
        return int(ordered[0]) if ordered else None

    def apply_cross_channel_overlap_settings(self, settings=None):
        self.overlap_review_settings = self._normalize_overlap_review_settings(settings)
        count = len(self.starts)
        self.overlap_tagged = np.zeros(count, dtype=int)
        self.overlap_group_ids = np.full(count, -1, dtype=int)
        self.overlap_channel_counts = np.zeros(count, dtype=int)
        self.overlap_event_counts = np.zeros(count, dtype=int)
        self.overlap_visibility_mask = np.ones(count, dtype=bool)

        if count == 0 or self.overlap_review_settings["action"] == self.OVERLAP_ACTION_DISABLED:
            self._ensure_current_visible_index()
            self._increment_view_state_token()
            return self.get_overlap_review_summary()

        groups = self._find_overlap_components(
            self.overlap_review_settings["min_overlap_ms"],
            self.overlap_review_settings["min_channels"],
        )
        for group_id, raw_indexes in enumerate(groups):
            channel_count = len({str(self.channel_names[raw_index]) for raw_index in raw_indexes})
            event_count = len(raw_indexes)
            primary_index = self._select_group_primary_index(raw_indexes)
            for raw_index in raw_indexes:
                self.overlap_group_ids[raw_index] = group_id
                self.overlap_channel_counts[raw_index] = channel_count
                self.overlap_event_counts[raw_index] = event_count
                if raw_index != primary_index:
                    self.overlap_tagged[raw_index] = 1

        if self.overlap_review_settings["action"] == self.OVERLAP_ACTION_HIDE:
            self.overlap_visibility_mask = np.array(self.overlap_tagged < 1, dtype=bool)

        self._ensure_current_visible_index()
        self._increment_view_state_token()
        return self.get_overlap_review_summary()

    def clear_cross_channel_overlap_settings(self):
        return self.apply_cross_channel_overlap_settings(self.default_overlap_review_settings())

    def _ensure_current_visible_index(self):
        visible_raw_indexes = self.get_visible_raw_indices()
        if len(visible_raw_indexes) == 0:
            self.index = 0
            return None
        if np.any(visible_raw_indexes == int(self.index)):
            return int(self.index)
        next_visible = visible_raw_indexes[visible_raw_indexes >= int(self.index)]
        self.index = int(next_visible[0] if len(next_visible) else visible_raw_indexes[0])
        return int(self.index)

    def get_current_visible_position(self):
        visible_raw_indexes = self.get_visible_raw_indices()
        if len(visible_raw_indexes) == 0:
            return 0
        current_raw_index = self._ensure_current_visible_index()
        positions = np.where(visible_raw_indexes == current_raw_index)[0]
        return int(positions[0]) if len(positions) else 0

    def _raw_index_from_visible_position(self, visible_index):
        visible_raw_indexes = self.get_visible_raw_indices()
        if len(visible_raw_indexes) == 0:
            return None
        visible_index = int(max(0, min(int(visible_index), len(visible_raw_indexes) - 1)))
        return int(visible_raw_indexes[visible_index])

    def _visible_position_from_raw_index(self, raw_index):
        visible_raw_indexes = self.get_visible_raw_indices()
        if len(visible_raw_indexes) == 0:
            return None
        positions = np.where(visible_raw_indexes == int(raw_index))[0]
        if len(positions) == 0:
            return None
        return int(positions[0])

    def has_prediction(self):
        return self.artifact_predicted

    def doctor_annotation(self, annotation: str):
        if self.get_raw_event_count() == 0:
            return
        self.clear_annotation(self.index)
        if annotation == "Artifact":
            self.artifact_annotations[self.index] = 1
        elif annotation == "Pathological":
            self.pathological_annotations[self.index] = 1
        elif annotation == "Physiological":
            self.physiological_annotations[self.index] = 1
        self.annotated[self.index] = 1

    def clear_annotation(self, index=None):
        if self.get_raw_event_count() == 0:
            return
        if index is None:
            index = self.index
        self.artifact_annotations[index] = 0
        self.pathological_annotations[index] = 0
        self.physiological_annotations[index] = 0
        self.annotated[index] = 0

    def _set_current_visible_position(self, visible_position):
        raw_index = self._raw_index_from_visible_position(visible_position)
        if raw_index is None:
            return None
        self.index = int(raw_index)
        return self.get_current()

    def get_next(self):
        visible_raw_indexes = self.get_visible_raw_indices()
        if len(visible_raw_indexes) == 0:
            return None
        current_pos = self.get_current_visible_position()
        next_pos = (current_pos + 1) % len(visible_raw_indexes)
        return self._set_current_visible_position(next_pos)

    def get_prev(self):
        visible_raw_indexes = self.get_visible_raw_indices()
        if len(visible_raw_indexes) == 0:
            return None
        current_pos = self.get_current_visible_position()
        prev_pos = (current_pos - 1) % len(visible_raw_indexes)
        return self._set_current_visible_position(prev_pos)

    def get_jump(self, index):
        return self._set_current_visible_position(index)

    def get_current(self):
        current_raw_index = self._ensure_current_visible_index()
        if current_raw_index is None:
            return None
        return (
            self.channel_names[current_raw_index],
            self.starts[current_raw_index],
            self.ends[current_raw_index],
        )

    def _get_prediction(self, predictions):
        artifact_val = predictions.get("Artifact", 1)
        if artifact_val == -1:
            return "Unpredicted"
        if artifact_val != 1:
            return "Artifact"

        priority_order = ["spkHFO", "eHFO", "HFO"]
        detected_labels = [label for label in priority_order if predictions.get(label, 0) == 1]
        return " and ".join(detected_labels) if detected_labels else "HFO"

    def _get_annotation(self, artifact_annotation, pathological_annotation, physiological_annotation):
        if artifact_annotation == 1:
            return "Artifact"
        if pathological_annotation == 1:
            return "Pathological"
        if physiological_annotation == 1:
            return "Physiological"
        return None

    def get_annotation(self, index=None):
        if index is None:
            index = self.index
        if len(self.annotated) == 0 or self.annotated[index] == 0:
            return None
        return self._get_annotation(
            self.artifact_annotations[index],
            self.pathological_annotations[index],
            self.physiological_annotations[index],
        )

    def _find_unannotated_index(self, direction=1):
        visible_raw_indexes = self.get_visible_raw_indices()
        if len(visible_raw_indexes) == 0 or np.all(self.annotated[visible_raw_indexes] > 0):
            return None
        current_pos = self.get_current_visible_position()
        for offset in range(1, len(visible_raw_indexes) + 1):
            candidate = visible_raw_indexes[(current_pos + direction * offset) % len(visible_raw_indexes)]
            if self.annotated[candidate] == 0:
                return int(candidate)
        return None

    def get_next_unannotated(self):
        next_index = self._find_unannotated_index(direction=1)
        if next_index is None:
            return None
        self.index = next_index
        return self.get_current()

    def get_prev_unannotated(self):
        prev_index = self._find_unannotated_index(direction=-1)
        if prev_index is None:
            return None
        self.index = prev_index
        return self.get_current()

    def get_prediction_scope_options(self):
        options = ["All"]
        if self.artifact_predicted:
            options.extend(["Artifact", "Non-artifact"])
            if self.spike_predicted and len(self.spike_predictions) == self.get_raw_event_count():
                options.append("spkHFO")
            if self.ehfo_predicted and len(self.ehfo_predictions) == self.get_raw_event_count():
                options.append("eHFO")
        if np.any(self.get_visible_array("overlap_tagged") > 0):
            options.append("Overlap tagged")
        return options

    def _matches_prediction_scope(self, index, scope):
        if scope in (None, "", "All"):
            return True
        if index < 0 or index >= self.get_raw_event_count():
            return False
        if scope == "Overlap tagged":
            return len(self.overlap_tagged) > index and self.overlap_tagged[index] > 0
        if not self.artifact_predicted:
            return False

        artifact_prediction = self.artifact_predictions[index]
        if scope == "Artifact":
            return artifact_prediction < 1
        if scope == "Non-artifact":
            return artifact_prediction > 0
        if artifact_prediction < 1:
            return False
        if scope == "spkHFO":
            return len(self.spike_predictions) > index and self.spike_predictions[index] == 1
        if scope == "eHFO":
            return len(self.ehfo_predictions) > index and self.ehfo_predictions[index] == 1
        return False

    def get_matching_indexes(self, scope="All", unannotated_only=False):
        visible_raw_indexes = self.get_visible_raw_indices()
        if len(visible_raw_indexes) == 0:
            return np.array([], dtype=int)
        indexes = [
            raw_index
            for raw_index in visible_raw_indexes
            if self._matches_prediction_scope(int(raw_index), scope)
            and (not unannotated_only or self.annotated[raw_index] == 0)
        ]
        return np.array(indexes, dtype=int)

    def _find_matching_index(self, scope="All", direction=1, unannotated_only=False):
        visible_raw_indexes = self.get_visible_raw_indices()
        matches = self.get_matching_indexes(scope, unannotated_only=unannotated_only)
        if len(visible_raw_indexes) == 0 or len(matches) == 0:
            return None
        match_set = set(matches.tolist())
        current_pos = self.get_current_visible_position()
        for offset in range(1, len(visible_raw_indexes) + 1):
            candidate = int(visible_raw_indexes[(current_pos + direction * offset) % len(visible_raw_indexes)])
            if candidate in match_set:
                return candidate
        return None

    def get_next_matching(self, scope="All", unannotated_only=False):
        next_index = self._find_matching_index(scope, direction=1, unannotated_only=unannotated_only)
        if next_index is None:
            return None
        self.index = next_index
        return self.get_current()

    def get_prev_matching(self, scope="All", unannotated_only=False):
        prev_index = self._find_matching_index(scope, direction=-1, unannotated_only=unannotated_only)
        if prev_index is None:
            return None
        self.index = prev_index
        return self.get_current()

    def get_review_progress(self):
        visible_raw_indexes = self.get_visible_raw_indices()
        reviewed = int(np.sum(self.annotated[visible_raw_indexes] > 0)) if len(visible_raw_indexes) else 0
        total = self.get_visible_event_count()
        return {"reviewed": reviewed, "remaining": max(total - reviewed, 0), "total": total}

    def get_annotation_counts(self):
        visible_raw_indexes = self.get_visible_raw_indices()
        if len(visible_raw_indexes) == 0:
            return {"Artifact": 0, "Pathological": 0, "Physiological": 0}
        return {
            "Artifact": int(np.sum(self.artifact_annotations[visible_raw_indexes] > 0)),
            "Pathological": int(np.sum(self.pathological_annotations[visible_raw_indexes] > 0)),
            "Physiological": int(np.sum(self.physiological_annotations[visible_raw_indexes] > 0)),
        }

    def has_reviewable_events(self):
        return self.get_visible_event_count() > 0

    def _get_overlap_tag_for_raw_index(self, raw_index):
        if len(self.overlap_tagged) <= raw_index or self.overlap_tagged[raw_index] == 0:
            return None
        return self.overlap_review_settings.get("tag_name", self.DEFAULT_OVERLAP_TAG)

    def get_current_info(self):
        current = self.get_current()
        if current is None:
            return {
                "channel_name": "",
                "start_index": 0,
                "end_index": 0,
                "prediction": None,
                "annotation": None,
                "overlap_tag": None,
                "display_index": 0,
            }

        current_raw_index = self.index
        prediction = None
        if self.artifact_predicted and len(self.artifact_predictions) > current_raw_index:
            spike_prediction = self.spike_predictions[current_raw_index] if len(self.spike_predictions) > current_raw_index else 0
            ehfo_prediction = self.ehfo_predictions[current_raw_index] if len(self.ehfo_predictions) > current_raw_index else 0
            prediction = self._get_prediction(
                {
                    "Artifact": self.artifact_predictions[current_raw_index],
                    "spkHFO": spike_prediction,
                    "eHFO": ehfo_prediction,
                }
            )

        return {
            "channel_name": current[0],
            "start_index": current[1],
            "end_index": current[2],
            "prediction": prediction,
            "annotation": self.get_annotation(current_raw_index),
            "overlap_tag": self._get_overlap_tag_for_raw_index(current_raw_index),
            "display_index": self.get_current_visible_position(),
            "display_total": self.get_visible_event_count(),
        }

    def get_num_artifact(self):
        visible_predictions = self.get_visible_array("artifact_predictions")
        if len(visible_predictions) == 0:
            return 0
        return int(np.sum(visible_predictions < 1))

    def get_num_spike(self):
        visible_predictions = self.get_visible_array("spike_predictions")
        if len(visible_predictions) == 0:
            return 0
        return int(np.sum(visible_predictions == 1))

    def get_num_ehfo(self):
        visible_predictions = self.get_visible_array("ehfo_predictions")
        if len(visible_predictions) == 0:
            return 0
        return int(np.sum(visible_predictions == 1))

    def get_num_real(self):
        visible_predictions = self.get_visible_array("artifact_predictions")
        if len(visible_predictions) == 0:
            return 0
        return int(np.sum(visible_predictions > 0))

    def has_feature(self):
        return len(self.features) > 0

    def get_features(self):
        return self.features

    def get_raw_spectrums(self):
        return self.raw_spectrums

    def to_dict(self):
        return {
            "channel_names": self.channel_names,
            "starts": self.starts,
            "ends": self.ends,
            "artifact_predictions": np.array(self.artifact_predictions),
            "spike_predictions": np.array(self.spike_predictions),
            "ehfo_predictions": np.array(self.ehfo_predictions),
            "feature": self.features,
            "HFO_type": self.HFO_type,
            "sample_freq": self.sample_freq,
            "feature_size": self.feature_size,
            "freq_range": self.freq_range,
            "time_range": self.time_range,
            "overlap_review_settings": dict(self.overlap_review_settings),
            "overlap_tagged": np.array(self.overlap_tagged),
            "overlap_group_ids": np.array(self.overlap_group_ids),
            "overlap_channel_counts": np.array(self.overlap_channel_counts),
            "overlap_event_counts": np.array(self.overlap_event_counts),
            "overlap_visibility_mask": np.array(self.overlap_visibility_mask, dtype=bool),
        }

    @staticmethod
    def from_dict(data):
        """
        construct HFO_Feature object from dictionary
        """
        channel_names = data["channel_names"]
        starts = data["starts"]
        ends = data["ends"]
        artifact_predictions = data["artifact_predictions"]
        spike_predictions = data["spike_predictions"]
        ehfo_predictions = data["ehfo_predictions"]
        feature = data["feature"]
        HFO_type = data["HFO_type"]
        sample_freq = data["sample_freq"]
        feature_size = data["feature_size"]
        freq_range = data["freq_range"]
        time_range = data["time_range"]
        biomarker_feature = HFO_Feature(
            channel_names,
            np.array([starts, ends]).T,
            feature,
            HFO_type,
            sample_freq,
            freq_range,
            time_range,
            feature_size,
        )
        biomarker_feature.update_pred(artifact_predictions, spike_predictions, ehfo_predictions)
        biomarker_feature.overlap_review_settings = biomarker_feature._normalize_overlap_review_settings(
            data.get("overlap_review_settings")
        )
        biomarker_feature.overlap_tagged = np.array(data.get("overlap_tagged", np.zeros(len(starts), dtype=int)))
        biomarker_feature.overlap_group_ids = np.array(data.get("overlap_group_ids", np.full(len(starts), -1, dtype=int)))
        biomarker_feature.overlap_channel_counts = np.array(data.get("overlap_channel_counts", np.zeros(len(starts), dtype=int)))
        biomarker_feature.overlap_event_counts = np.array(data.get("overlap_event_counts", np.zeros(len(starts), dtype=int)))
        biomarker_feature.overlap_visibility_mask = np.array(
            data.get("overlap_visibility_mask", np.ones(len(starts), dtype=bool))
        )
        biomarker_feature._repair_overlap_view_state()
        return biomarker_feature

    def update_artifact_pred(self, artifact_predictions):
        self.artifact_predicted = True
        self.artifact_predictions = np.array(artifact_predictions)
        self.num_artifact = int(np.sum(self.artifact_predictions < 1))
        self.num_real = int(np.sum(self.artifact_predictions > 0))

    def update_spike_pred(self, spike_predictions):
        self.spike_predicted = True
        self.spike_predictions = np.array(spike_predictions)
        self.num_spike = int(np.sum(self.spike_predictions == 1))

    def update_ehfo_pred(self, ehfo_predictions):
        self.ehfo_predicted = True
        self.ehfo_predictions = np.array(ehfo_predictions)
        self.num_ehfo = int(np.sum(self.ehfo_predictions == 1))

    def update_pred(self, artifact_predictions, spike_predictions, ehfo_predictions):
        artifact_predictions = np.array(artifact_predictions)
        if len(artifact_predictions) > 0:
            has_actual_predictions = np.any((artifact_predictions != 0) & (artifact_predictions != -1))
            if has_actual_predictions:
                self.update_artifact_pred(artifact_predictions)
            else:
                self.artifact_predictions = artifact_predictions
                self.artifact_predicted = False
        else:
            self.artifact_predictions = artifact_predictions
            self.artifact_predicted = False
        self.update_spike_pred(spike_predictions)
        self.update_ehfo_pred(ehfo_predictions)

    def group_by_channel(self):
        channel_names = self.get_visible_array("channel_names")
        starts = self.get_visible_array("starts")
        ends = self.get_visible_array("ends")
        artifact_predictions = np.array(self.get_visible_array("artifact_predictions"))
        spike_predictions = np.array(self.get_visible_array("spike_predictions"))
        channel_names_unique = np.unique(channel_names)
        interval = np.array([starts, ends]).T if len(starts) else np.empty((0, 2), dtype=int)
        channel_name_g, interval_g, artifact_predictions_g, spike_predictions_g = [], [], [], []
        for channel_name in channel_names_unique:
            channel_index = np.where(channel_names == channel_name)[0]
            interval_g.append(interval[channel_index])
            channel_name_g.append(channel_name)
            if len(artifact_predictions) > 0:
                artifact_predictions_g.append(artifact_predictions[channel_index])
            if len(spike_predictions) > 0:
                spike_predictions_g.append(spike_predictions[channel_index])
        return channel_name_g, interval_g, artifact_predictions_g, spike_predictions_g

    def _build_channel_index_cache(self):
        if self._channel_index_cache is not None:
            return self._channel_index_cache
        cache = {}
        for idx, channel_name in enumerate(self.channel_names):
            cache.setdefault(channel_name, []).append(idx)
        self._channel_index_cache = {
            channel_name: np.array(indexes, dtype=int) for channel_name, indexes in cache.items()
        }
        return self._channel_index_cache

    def get_biomarkers_for_channel(self, channel_name: str, min_start: int = None, max_end: int = None):
        channel_indexes = self._build_channel_index_cache().get(channel_name, np.array([], dtype=int))
        visible_indexes = self.get_visible_raw_indices()
        if len(channel_indexes) and len(visible_indexes):
            channel_indexes = np.intersect1d(channel_indexes, visible_indexes, assume_unique=False)
        else:
            channel_indexes = np.array([], dtype=int)

        starts = self.starts[channel_indexes]
        ends = self.ends[channel_indexes]
        artifact_predictions = np.array(self.artifact_predictions)
        spike_predictions = np.array(self.spike_predictions)
        ehfo_predictions = np.array(self.ehfo_predictions)
        if min_start is not None and max_end is not None:
            # Keep events that overlap the visible window so zooming does not
            # hide markers sitting on a window edge.
            window_indexes = (starts < max_end) & (ends > min_start)
            starts = starts[window_indexes]
            ends = ends[window_indexes]
            channel_indexes = channel_indexes[window_indexes]
        try:
            artifact_predictions = artifact_predictions[channel_indexes]
            spike_predictions = spike_predictions[channel_indexes] == 1
            ehfo_predictions = ehfo_predictions[channel_indexes] == 1
        except (IndexError, TypeError):
            artifact_predictions = []
            spike_predictions = []
            ehfo_predictions = []
        return starts, ends, artifact_predictions, spike_predictions, ehfo_predictions

    def get_annotation_text(self, index, raw_index=False):
        raw_event_index = int(index) if raw_index else self._raw_index_from_visible_position(index)
        if raw_event_index is None:
            return " No visible events"

        channel_name = self.channel_names[raw_event_index]
        if self.annotated[raw_event_index] == 0:
            suffix = "Unannotated"
        elif self.artifact_annotations[raw_event_index] == 1:
            suffix = "Artifact"
        elif self.pathological_annotations[raw_event_index] == 1:
            suffix = "Pathological"
        elif self.physiological_annotations[raw_event_index] == 1:
            suffix = "Physiological"
        else:
            suffix = "Unknown"

        label = f" No.{(self._visible_position_from_raw_index(raw_event_index) or 0) + 1}: {channel_name} : {suffix}"
        overlap_tag = self._get_overlap_tag_for_raw_index(raw_event_index)
        if overlap_tag:
            label += f" [{overlap_tag}]"
        return label

    def to_df(self):
        visible_raw_indexes = self.get_visible_raw_indices()
        df = pd.DataFrame()
        if len(visible_raw_indexes) == 0:
            if self.overlap_review_settings.get("action") != self.OVERLAP_ACTION_DISABLED:
                df["overlap review action"] = pd.Series(dtype=object)
            return df

        artifact_predictions = np.array(self.artifact_predictions)
        spike_predictions = np.array(self.spike_predictions)
        ehfo_predictions = np.array(self.ehfo_predictions)
        artifact_annotations = np.array(self.artifact_annotations)
        pathological_annotations = np.array(self.pathological_annotations)
        physiological_annotations = np.array(self.physiological_annotations)
        annotated = np.array(self.annotated)

        df["channel_names"] = self.channel_names[visible_raw_indexes]
        df["starts"] = self.starts[visible_raw_indexes]
        df["ends"] = self.ends[visible_raw_indexes]
        if len(artifact_predictions) > 0:
            df["artifact"] = artifact_predictions[visible_raw_indexes]
        if len(spike_predictions) > 0:
            df["spike"] = spike_predictions[visible_raw_indexes]
        if len(ehfo_predictions) > 0:
            df["ehfo"] = ehfo_predictions[visible_raw_indexes]
        df["annotated"] = annotated[visible_raw_indexes]
        if len(artifact_annotations) > 0:
            df["artifact annotations"] = artifact_annotations[visible_raw_indexes]
        if len(pathological_annotations) > 0:
            df["pathological annotations"] = pathological_annotations[visible_raw_indexes]
        if len(physiological_annotations) > 0:
            df["physiological annotations"] = physiological_annotations[visible_raw_indexes]

        overlap_action = self.overlap_review_settings.get("action", self.OVERLAP_ACTION_DISABLED)
        if overlap_action != self.OVERLAP_ACTION_DISABLED:
            df["overlap review action"] = overlap_action
            df["overlap strategy"] = "keep_first"
            df["review tags"] = [
                self._get_overlap_tag_for_raw_index(raw_index) or ""
                for raw_index in visible_raw_indexes
            ]
            df["overlap tagged"] = self.overlap_tagged[visible_raw_indexes]
            df["overlap primary"] = (
                (self.overlap_group_ids[visible_raw_indexes] >= 0)
                & (self.overlap_tagged[visible_raw_indexes] < 1)
            ).astype(int)
            df["overlap group id"] = np.where(self.overlap_group_ids[visible_raw_indexes] >= 0, self.overlap_group_ids[visible_raw_indexes] + 1, 0)
            df["overlap channels"] = self.overlap_channel_counts[visible_raw_indexes]
            df["overlap events"] = self.overlap_event_counts[visible_raw_indexes]
        return df

    def export_csv(self, file_path):
        self.to_df().to_csv(file_path, index=False)

    def export_excel(self, file_path):
        df = self.to_df()
        df_out = df.copy()
        if "artifact" not in df_out.columns:
            df_out["artifact"] = 0
        if "spike" not in df_out.columns:
            df_out["spike"] = 0
        if "ehfo" not in df_out.columns:
            df_out["ehfo"] = 0
        if "pathological annotations" not in df_out.columns:
            df_out["pathological annotations"] = 0
        if "physiological annotations" not in df_out.columns:
            df_out["physiological annotations"] = 0
        if "artifact annotations" not in df_out.columns:
            df_out["artifact annotations"] = 0
        df_out["artifact"] = (df_out["artifact"] > 0).astype(int)
        df_out["spike"] = (df_out["spike"] > 0).astype(int)
        df_out["ehfo"] = (df_out["ehfo"] > 0).astype(int)
        df_out["annotated"] = 1 - (df_out["annotated"] > 0).astype(int)
        df_out["artifact annotations"] = (df_out["artifact annotations"] > 0).astype(int)
        df_out["pathological annotations"] = (df_out["pathological annotations"] > 0).astype(int)
        df_out["physiological annotations"] = (df_out["physiological annotations"] > 0).astype(int)

        df_channel = df_out.groupby("channel_names").agg(
            {
                "starts": "count",
                "artifact": "sum",
                "spike": "sum",
                "ehfo": "sum",
                "annotated": "sum",
                "artifact annotations": "sum",
                "pathological annotations": "sum",
                "physiological annotations": "sum",
            }
        ).reset_index()
        df_channel.rename(
            columns={
                "starts": "Total Detection",
                "artifact": "HFO",
                "spike": "spkHFO",
                "ehfo": "eHFO",
                "annotated": "Unannotated",
                "artifact annotations": "HFO annotations",
            },
            inplace=True,
        )
        df.rename(
            columns={
                "artifact": "HFO",
                "spike": "spkHFO",
                "ehfo": "eHFO",
                "annotated": "Annotated",
                "artifact annotations": "HFO annotations",
            },
            inplace=True,
        )
        if "Annotated" in df.columns:
            df["Annotated"] = df["Annotated"] > 0
            df["Annotated"] = df["Annotated"].replace({True: "Yes", False: "No"})
        with pd.ExcelWriter(file_path) as writer:
            df_channel.to_excel(writer, sheet_name="Channels", index=False)
            df.to_excel(writer, sheet_name="Events", index=False)
