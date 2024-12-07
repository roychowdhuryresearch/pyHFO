import numpy as np
import pandas as pd


class SpindleFeature(object):
    def __init__(self, channel_names, starts, ends, features=[], detector_type="STE", sample_freq=2000, freq_range=[10, 500],
                 time_range=[0, 1000], feature_size=224):
        self.channel_names = channel_names
        if starts.size == 0:
            self.starts = np.array([])
            self.ends = np.array([])
            self.artifact_predictions = np.array([])
            self.artifact_annotations = np.array([])
            self.spike_annotations = np.array([])
            self.annotated = np.array([])
        else:
            self.starts = starts
            self.ends = ends
            self.features = features
            self.artifact_predictions = np.zeros(self.starts.shape)
            self.spike_predictions = []
            self.artifact_annotations = np.zeros(self.starts.shape)
            self.spike_annotations = np.zeros(self.starts.shape)
            self.annotated = np.zeros(self.starts.shape)
        self.detector_type = detector_type
        self.sample_freq = sample_freq
        self.feature_size = 0
        self.freq_range = freq_range
        self.time_range = time_range
        self.feature_size = feature_size
        self.num_artifact = 0
        self.num_spike = 0
        self.num_spindle = len(self.starts)
        self.num_real = 0
        self.index = 0
        self.artifact_predicted = False
        self.spike_predicted = False

    def __str__(self):
        return "Spindle_Feature: {} Spindles, {} artifacts, {} spikes, {} real Spindles".format(self.num_spindle, self.num_artifact,
                                                                                    self.num_spike, self.num_real)

    @staticmethod
    def construct(result, detector_type="STE", sample_freq=2000, freq_range=[10, 500],
                  time_range=[0, 1000], feature_size=224):
        '''
        Construct SpindleFeature object from detector output
        '''
        result_summary = result.summary()
        channel_names = result_summary['Channel'].to_numpy()
        start = result_summary['Start'].to_numpy() * sample_freq
        end = result_summary['End'].to_numpy() * sample_freq
        return SpindleFeature(channel_names, start, end, np.array([]), detector_type, sample_freq, freq_range, time_range, feature_size)

    def get_num_biomarker(self):
        return self.num_spindle

    def has_prediction(self):
        return self.artifact_predicted

    # def generate_psedo_label(self):
    #     self.artifact_predictions = np.ones(self.num_spindle)
    #     self.spike_predictions = np.zeros(self.num_spindle)
    #     self.artifact_predicted = True

    def doctor_annotation(self, annotation: str):
        if annotation == "Artifact":
            self.artifact_annotations[self.index] = 0
        elif annotation == "Spike":
            self.spike_annotations[self.index] = 1
            self.artifact_annotations[self.index] = 1
        elif annotation == "Real":
            self.spike_annotations[self.index] = 0
            self.artifact_annotations[self.index] = 1
        self.annotated[self.index] = 1

    def get_next(self):
        if self.index >= self.num_spindle - 1:
            self.index = 0
        else:
            self.index += 1
        # returns the next hfo start and end index instead of next window start and end index
        return self.channel_names[self.index], self.starts[self.index], self.ends[self.index]

    def get_prev(self):
        if self.index <= 0:
            self.index = 0
        else:
            self.index -= 1
        # the same as above
        return self.channel_names[self.index], self.starts[self.index], self.ends[self.index]

    def get_jump(self, index):
        self.index = index
        # the same as above
        return self.channel_names[self.index], self.starts[self.index], self.ends[self.index]

    def get_current(self):
        return self.channel_names[self.index], self.starts[self.index], self.ends[self.index]

    def _get_prediction(self, artifact_prediction, spike_prediction):
        if artifact_prediction < 1:
            return "Artifact"
        elif spike_prediction == 1:
            return "Spike"
        else:
            return "Spindle"

    def get_current_info(self):
        print("self.artifact_predicted:", self.artifact_predicted)
        channel_name = self.channel_names[self.index]
        start = self.starts[self.index]
        end = self.ends[self.index]
        prediction = self._get_prediction(self.artifact_predictions[self.index],
                                          self.spike_predictions[self.index]) if self.artifact_predicted else None
        annotation = self._get_prediction(self.artifact_annotations[self.index], self.spike_annotations[self.index]) if \
        self.annotated[self.index] else None
        return {"channel_name": channel_name, "start_index": start, "end_index": end, "prediction": prediction,
                "annotation": annotation}

    def get_num_artifact(self):
        return self.num_artifact

    def get_num_spike(self):
        return self.num_spike

    def get_num_real(self):
        return self.num_real

    def has_feature(self):
        return len(self.features) > 0

    def get_features(self):
        return self.features

    def to_dict(self):
        channel_names = self.channel_names
        starts = self.starts
        ends = self.ends
        artifact_predictions = np.array(self.artifact_predictions)
        spike_predictions = np.array(self.spike_predictions)
        feature = self.features
        detector_type = self.detector_type
        sample_freq = self.sample_freq
        feature_size = self.feature_size
        freq_range = self.freq_range
        time_range = self.time_range
        return {"channel_names": channel_names, "starts": starts, "ends": ends,
                "artifact_predictions": artifact_predictions, "spike_predictions": spike_predictions,
                "feature": feature, "detector_type": detector_type, "sample_freq": sample_freq, "feature_size": feature_size,
                "freq_range": freq_range, "time_range": time_range}

    @staticmethod
    def from_dict(data):
        '''
        construct SpindleFeature object from dictionary
        '''
        channel_names = data["channel_names"]
        starts = data["starts"]
        ends = data["ends"]
        artifact_predictions = data["artifact_predictions"]
        spike_predictions = data["spike_predictions"]
        feature = data["feature"]
        detector_type = data["detector_type"]
        sample_freq = data["sample_freq"]
        feature_size = data["feature_size"]
        freq_range = data["freq_range"]
        time_range = data["time_range"]
        biomarker_feature = SpindleFeature(channel_names, np.array([starts, ends]).T, feature, detector_type, sample_freq, freq_range,
                                  time_range, feature_size)
        biomarker_feature.update_pred(artifact_predictions, spike_predictions)

        return biomarker_feature

    def update_artifact_pred(self, artifact_predictions):
        self.artifact_predicted = True
        self.artifact_predictions = artifact_predictions
        self.num_artifact = np.sum(artifact_predictions < 1)
        self.num_real = np.sum(artifact_predictions > 0)

    def update_spike_pred(self, spike_predictions):
        self.spike_predicted = True
        self.spike_predictions = spike_predictions
        self.num_spike = np.sum(spike_predictions == 1)

    def update_pred(self, artifact_predictions, spike_predictions):
        self.update_artifact_pred(artifact_predictions)
        self.update_spike_pred(spike_predictions)

    def group_by_channel(self):
        channel_names = self.channel_names
        starts = self.starts
        ends = self.ends
        artifact_predictions = np.array(self.artifact_predictions)
        spike_predictions = np.array(self.spike_predictions)
        channel_names_unique = np.unique(channel_names)
        interval = np.array([starts, ends]).T
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

    def get_biomarkers_for_channel(self, channel_name: str, min_start: int = None, max_end: int = None):
        channel_names = self.channel_names
        starts = self.starts
        ends = self.ends
        artifact_predictions = np.array(self.artifact_predictions)
        spike_predictions = np.array(self.spike_predictions)
        indexes = channel_names == channel_name
        if min_start is not None and max_end is not None:
            indexes = indexes & (starts >= min_start) & (ends <= max_end)
        starts = starts[indexes]
        ends = ends[indexes]
        try:
            artifact_predictions = artifact_predictions[indexes]
            spike_predictions = spike_predictions[indexes] == 1
        except:
            artifact_predictions = []
            spike_predictions = []
        return starts, ends, artifact_predictions, spike_predictions

    def get_annotation_text(self, index):
        channel_name = self.channel_names[index]
        if self.annotated[index] == 0:
            suffix = "Unannotated"
        elif self.artifact_annotations[index] == 0:
            suffix = "Artifact"
        elif self.spike_annotations[index] == 1:
            suffix = "Spike"
        else:
            suffix = "Real"
        return f" No.{index + 1}: {channel_name} : {suffix}"

    def to_df(self):
        channel_names = self.channel_names
        starts = self.starts
        ends = self.ends
        artifact_predictions = np.array(self.artifact_predictions)
        spike_predictions = np.array(self.spike_predictions)
        artifact_annotations = np.array(self.artifact_annotations)
        spike_annotations = np.array(self.spike_annotations)
        annotated = np.array(self.annotated)
        df = pd.DataFrame()
        df["channel_names"] = channel_names
        df["starts"] = starts
        df["ends"] = ends
        # df["doctor_annotation"] = self.doctor_annotation
        if len(artifact_predictions) > 0:
            df["artifact"] = artifact_predictions
        if len(spike_predictions) > 0:
            df["spike"] = spike_predictions
        df['annotated'] = annotated
        if len(artifact_annotations) > 0:
            df["artifact annotations"] = artifact_annotations
        if len(spike_annotations) > 0:
            df["spike annotations"] = spike_annotations
        return df

    def export_csv(self, file_path):
        df = self.to_df()
        df.to_csv(file_path, index=False)

    def export_excel(self, file_path):
        df = self.to_df()
        df_out = df.copy()
        if "artifact" not in df_out.columns:
            df_out["artifact"] = 0
        if "spike" not in df_out.columns:
            df_out["spike"] = 0
        if "artifact annotations" not in df_out.columns:
            df_out["artifact annotations"] = 0
        if "spike annotations" not in df_out.columns:
            df_out["spike annotations"] = 0
        df_out["artifact"] = (df_out["artifact"] > 0).astype(int)
        df_out["spike"] = (df_out["spike"] > 0).astype(int)
        df_out['annotated'] = 1 - (df_out["annotated"] > 0).astype(int)
        df_out["artifact annotations"] = (df_out["artifact annotations"] > 0).astype(int)
        df_out["spike annotations"] = (df_out["spike annotations"] > 0).astype(int)
        df_channel = df_out.groupby("channel_names").agg({"starts": "count",
                                                          "artifact": "sum", "spike": "sum",
                                                          "annotated": "sum",
                                                          "artifact annotations": "sum",
                                                          "spike annotations": "sum"}).reset_index()
        df_channel.rename(columns={"starts": "Total Detection",
                                   "artifact": "Spindle", "spike": "spk-Spindle",
                                   "annotated": "Unannotated",
                                   "artifact annotations": "Spindle annotations",
                                   "spike annotations": "spk-Spindle annotations"}, inplace=True)
        df.rename(columns={"artifact": "Spindle", "spike": "spk-Spindle",
                           "annotated": "Annotated",
                           "artifact annotations": "Spindle annotations", "spike annotations": "spk-Spindle annotations"},
                  inplace=True)
        df['Annotated'] = df["Annotated"] > 0
        df['Annotated'] = df['Annotated'].replace({True: 'Yes', False: 'No'})
        with pd.ExcelWriter(file_path) as writer:
            df_channel.to_excel(writer, sheet_name="Channels", index=False)
            df.to_excel(writer, sheet_name="Events", index=False)