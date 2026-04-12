import numpy as np
import pandas as pd


class SpikeFeature:
    def __init__(
        self,
        channel_names,
        starts,
        ends,
        features=[],
        detector_type="RMS/LL",
        sample_freq=2000,
        freq_range=[1, 80],
        time_range=[0, 1000],
        feature_size=224,
    ):
        self.channel_names = np.array(channel_names)
        self.features = features
        if self.channel_names.size == 0:
            self.starts = np.array([], dtype=int)
            self.ends = np.array([], dtype=int)
            self.artifact_predictions = np.array([])
            self.accepted_predictions = np.array([])
            self.artifact_annotations = np.array([])
            self.accepted_annotations = np.array([])
            self.annotated = np.array([])
        else:
            self.starts = np.array(starts, dtype=int)
            self.ends = np.array(ends, dtype=int)
            self.artifact_predictions = np.array([])
            self.accepted_predictions = np.array([])
            self.artifact_annotations = np.zeros(self.starts.shape, dtype=float)
            self.accepted_annotations = np.zeros(self.starts.shape, dtype=float)
            self.annotated = np.zeros(self.starts.shape, dtype=float)

        self.detector_type = detector_type
        self.sample_freq = sample_freq
        self.freq_range = freq_range
        self.time_range = time_range
        self.feature_size = feature_size
        self.num_spike = int(len(self.starts))
        self.num_artifact = 0
        self.num_real = 0
        self.index = 0
        self.artifact_predicted = False
        self.accepted_predicted = False
        self.spike_predicted = False
        self._channel_index_cache = None

    def __str__(self):
        return (
            "SpikeFeature: {} candidates, {} artifacts, {} accepted".format(
                self.get_num_biomarker(),
                self.get_num_artifact(),
                self.get_num_real(),
            )
        )

    @staticmethod
    def construct(
        channel_names,
        start_end,
        detector_type="RMS/LL",
        sample_freq=2000,
        freq_range=[1, 80],
        time_range=[0, 1000],
        feature_size=224,
    ):
        channel_names = np.asarray(channel_names)
        if len(channel_names) == 0 or len(start_end) == 0:
            return SpikeFeature(np.array([]), np.array([]), np.array([]), np.array([]), detector_type, sample_freq, freq_range, time_range, feature_size)

        expanded_channel_names = np.concatenate(
            [[channel_names[i]] * len(start_end[i]) for i in range(len(channel_names))]
        )
        intervals = [np.asarray(interval) for interval in start_end if len(interval) > 0]
        interval = np.concatenate(intervals) if intervals else np.empty((0, 2), dtype=int)

        if interval.size == 0:
            return SpikeFeature(np.array([]), np.array([]), np.array([]), np.array([]), detector_type, sample_freq, freq_range, time_range, feature_size)

        starts = np.asarray(interval[:, 0], dtype=int)
        ends = np.asarray(interval[:, 1], dtype=int)
        valid_mask = ends > starts
        return SpikeFeature(
            expanded_channel_names[valid_mask],
            starts[valid_mask],
            ends[valid_mask],
            np.array([]),
            detector_type,
            sample_freq,
            freq_range,
            time_range,
            feature_size,
        )

    def get_num_biomarker(self):
        return int(len(self.starts))

    def has_prediction(self):
        return self.artifact_predicted or self.accepted_predicted

    def doctor_annotation(self, annotation: str):
        if self.get_num_biomarker() == 0:
            return
        self.clear_annotation(self.index)
        if annotation == "Artifact":
            self.artifact_annotations[self.index] = 1
        elif annotation == "Accepted":
            self.accepted_annotations[self.index] = 1
        self.annotated[self.index] = 1
        self._refresh_annotation_counts()

    def clear_annotation(self, index=None):
        if self.get_num_biomarker() == 0:
            return
        if index is None:
            index = self.index
        self.artifact_annotations[index] = 0
        self.accepted_annotations[index] = 0
        self.annotated[index] = 0
        self._refresh_annotation_counts()

    def get_next(self):
        if self.get_num_biomarker() == 0:
            return None
        self.index = (self.index + 1) % self.get_num_biomarker()
        return self.get_current()

    def get_prev(self):
        if self.get_num_biomarker() == 0:
            return None
        self.index = (self.index - 1) % self.get_num_biomarker()
        return self.get_current()

    def get_jump(self, index):
        if self.get_num_biomarker() == 0:
            return None
        self.index = int(max(0, min(int(index), self.get_num_biomarker() - 1)))
        return self.get_current()

    def get_current(self):
        if self.get_num_biomarker() == 0:
            return None
        return self.channel_names[self.index], self.starts[self.index], self.ends[self.index]

    def get_annotation(self, index=None):
        if index is None:
            index = self.index
        if len(self.annotated) == 0 or self.annotated[index] == 0:
            return None
        if self.artifact_annotations[index] == 1:
            return "Artifact"
        if self.accepted_annotations[index] == 1:
            return "Accepted"
        return None

    def _find_unannotated_index(self, direction=1):
        if self.get_num_biomarker() == 0 or np.all(self.annotated > 0):
            return None
        for offset in range(1, self.get_num_biomarker() + 1):
            candidate = (self.index + direction * offset) % self.get_num_biomarker()
            if self.annotated[candidate] == 0:
                return candidate
        return None

    def get_next_unannotated(self):
        next_index = self._find_unannotated_index(direction=1)
        if next_index is None:
            return None
        return self.get_jump(next_index)

    def get_prev_unannotated(self):
        prev_index = self._find_unannotated_index(direction=-1)
        if prev_index is None:
            return None
        return self.get_jump(prev_index)

    def get_prediction_scope_options(self):
        return ["All"]

    def get_matching_indexes(self, scope="All", unannotated_only=False):
        if scope not in (None, "", "All"):
            return np.array([], dtype=int)
        indexes = [
            idx for idx in range(self.get_num_biomarker())
            if not unannotated_only or self.annotated[idx] == 0
        ]
        return np.array(indexes, dtype=int)

    def get_next_matching(self, scope="All", unannotated_only=False):
        matches = self.get_matching_indexes(scope, unannotated_only=unannotated_only)
        if len(matches) == 0:
            return None
        match_set = set(matches.tolist())
        for offset in range(1, self.get_num_biomarker() + 1):
            candidate = (self.index + offset) % self.get_num_biomarker()
            if candidate in match_set:
                return self.get_jump(candidate)
        return None

    def get_prev_matching(self, scope="All", unannotated_only=False):
        matches = self.get_matching_indexes(scope, unannotated_only=unannotated_only)
        if len(matches) == 0:
            return None
        match_set = set(matches.tolist())
        for offset in range(1, self.get_num_biomarker() + 1):
            candidate = (self.index - offset) % self.get_num_biomarker()
            if candidate in match_set:
                return self.get_jump(candidate)
        return None

    def get_review_progress(self):
        reviewed = int(np.sum(self.annotated > 0))
        total = self.get_num_biomarker()
        return {"reviewed": reviewed, "remaining": max(total - reviewed, 0), "total": total}

    def get_annotation_counts(self):
        return {
            "Artifact": int(np.sum(self.artifact_annotations > 0)),
            "Accepted": int(np.sum(self.accepted_annotations > 0)),
        }

    def has_reviewable_events(self):
        return self.get_num_biomarker() > 0

    def get_current_info(self):
        if self.get_num_biomarker() == 0:
            return {
                "channel_name": "",
                "start_index": 0,
                "end_index": 0,
                "prediction": None,
                "annotation": None,
            }
        annotation = self.get_annotation(self.index)
        prediction = annotation if annotation is not None else "Spike"
        return {
            "channel_name": self.channel_names[self.index],
            "start_index": self.starts[self.index],
            "end_index": self.ends[self.index],
            "prediction": prediction,
            "annotation": annotation,
        }

    def get_num_artifact(self):
        return self.num_artifact

    def get_num_spike(self):
        return self.num_spike

    def get_num_real(self):
        return self.num_real

    def _refresh_annotation_counts(self):
        self.num_artifact = int(np.sum(self.artifact_annotations > 0))
        self.num_real = int(np.sum(self.accepted_annotations > 0))

    def has_feature(self):
        return len(self.features) > 0

    def get_features(self):
        return self.features

    def to_dict(self):
        return {
            "channel_names": np.array(self.channel_names),
            "starts": np.array(self.starts),
            "ends": np.array(self.ends),
            "feature": self.features,
            "detector_type": self.detector_type,
            "sample_freq": self.sample_freq,
            "feature_size": self.feature_size,
            "freq_range": self.freq_range,
            "time_range": self.time_range,
            "artifact_predictions": np.array(self.artifact_predictions),
            "accepted_predictions": np.array(self.accepted_predictions),
            "artifact_annotations": np.array(self.artifact_annotations),
            "accepted_annotations": np.array(self.accepted_annotations),
            "annotated": np.array(self.annotated),
        }

    @staticmethod
    def from_dict(data):
        feature = SpikeFeature(
            data["channel_names"],
            data["starts"],
            data["ends"],
            data.get("feature", np.array([])),
            data.get("detector_type", "RMS/LL"),
            data.get("sample_freq", 2000),
            data.get("freq_range", [1, 80]),
            data.get("time_range", [0, 1000]),
            data.get("feature_size", 224),
        )
        feature.artifact_predictions = np.array(data.get("artifact_predictions", np.array([])))
        feature.accepted_predictions = np.array(data.get("accepted_predictions", np.array([])))
        feature.artifact_annotations = np.array(data.get("artifact_annotations", np.array([])))
        feature.accepted_annotations = np.array(data.get("accepted_annotations", np.array([])))
        feature.annotated = np.array(data.get("annotated", np.array([])))
        feature._refresh_annotation_counts()
        return feature

    def update_artifact_pred(self, artifact_predictions):
        self.artifact_predicted = True
        self.artifact_predictions = np.array(artifact_predictions)

    def update_accepted_pred(self, accepted_predictions):
        self.accepted_predicted = True
        self.spike_predicted = True
        self.accepted_predictions = np.array(accepted_predictions)

    def group_by_channel(self):
        channel_names = np.array(self.channel_names)
        starts = np.array(self.starts)
        ends = np.array(self.ends)
        channel_names_unique = np.unique(channel_names)
        interval = np.array([starts, ends]).T
        channel_name_g, interval_g = [], []
        for channel_name in channel_names_unique:
            channel_index = np.where(channel_names == channel_name)[0]
            interval_g.append(interval[channel_index])
            channel_name_g.append(channel_name)
        return channel_name_g, interval_g

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
        starts = self.starts[channel_indexes]
        ends = self.ends[channel_indexes]
        if min_start is not None and max_end is not None:
            indexes = (starts < max_end) & (ends > min_start)
            starts = starts[indexes]
            ends = ends[indexes]
            channel_indexes = channel_indexes[indexes]

        artifacts = np.ones(len(channel_indexes), dtype=int)
        spike_candidates = np.ones(len(channel_indexes), dtype=bool)
        for payload_index, raw_index in enumerate(channel_indexes):
            if self.annotated[raw_index] > 0 and self.accepted_annotations[raw_index] > 0:
                spike_candidates[payload_index] = False
            if self.annotated[raw_index] > 0 and self.artifact_annotations[raw_index] > 0:
                artifacts[payload_index] = 0
                spike_candidates[payload_index] = False
        return starts, ends, artifacts, spike_candidates

    def get_annotation_text(self, index):
        channel_name = self.channel_names[index]
        if self.annotated[index] == 0:
            suffix = "Spike"
        elif self.artifact_annotations[index] == 1:
            suffix = "Artifact"
        elif self.accepted_annotations[index] == 1:
            suffix = "Accepted"
        else:
            suffix = "Spike"
        return f" No.{index + 1}: {channel_name} : {suffix}"

    def to_df(self):
        df = pd.DataFrame()
        df["channel_names"] = np.array(self.channel_names)
        df["starts"] = np.array(self.starts)
        df["ends"] = np.array(self.ends)
        df["annotated"] = np.array(self.annotated)
        df["artifact annotations"] = np.array(self.artifact_annotations)
        df["accepted annotations"] = np.array(self.accepted_annotations)
        return df

    def export_csv(self, file_path):
        self.to_df().to_csv(file_path, index=False)

    def export_excel(self, file_path):
        df = self.to_df()
        df_out = df.copy()
        df_out["artifact annotations"] = (df_out["artifact annotations"] > 0).astype(int)
        df_out["accepted annotations"] = (df_out["accepted annotations"] > 0).astype(int)
        df_out["annotated"] = (df_out["annotated"] > 0).astype(int)
        df_out["unannotated"] = 1 - df_out["annotated"]

        df_channel = df_out.groupby("channel_names").agg(
            {
                "starts": "count",
                "accepted annotations": "sum",
                "artifact annotations": "sum",
                "unannotated": "sum",
            }
        ).reset_index()
        df_channel.rename(
            columns={
                "starts": "Total Detections",
                "accepted annotations": "Accepted spikes",
                "artifact annotations": "Artifact annotations",
                "unannotated": "Unannotated",
            },
            inplace=True,
        )

        df.rename(
            columns={
                "annotated": "Annotated",
                "accepted annotations": "Accepted annotations",
            },
            inplace=True,
        )
        df["Annotated"] = df["Annotated"] > 0
        df["Annotated"] = df["Annotated"].replace({True: "Yes", False: "No"})

        with pd.ExcelWriter(file_path) as writer:
            df_channel.to_excel(writer, sheet_name="Channels", index=False)
            df.to_excel(writer, sheet_name="Events", index=False)
