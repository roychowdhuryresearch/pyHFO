import numpy as np
import pandas as pd
class HFO_Feature():
    def __init__(self, channel_names, interval, features = [], HFO_type = "STE", sample_freq = 2000, freq_range = [10, 500], time_range = [0, 1000], feature_size = 224, raw_spectrums = None):
        self.channel_names = channel_names
        if interval.size == 0:
            self.starts = np.array([])
            self.ends = np.array([])
            self.artifact_predictions = np.array([])
            self.artifact_annotations = np.array([])
            # self.spike_annotations = np.array([])
            # self.ehfo_annotations = np.array([])
            self.pathological_annotations = np.array([])
            self.physiological_annotations = np.array([])
            self.annotated = np.array([])
        else:
            self.starts = interval[:, 0]
            self.ends = interval[:, 1]
            self.features = features
            self.artifact_predictions = np.zeros(self.starts.shape)
            self.artifact_annotations = np.zeros(self.starts.shape)
            self.spike_predictions = np.array([])
            self.ehfo_predictions = np.array([])
            self.pathological_annotations = np.zeros(self.starts.shape)
            self.physiological_annotations = np.zeros(self.starts.shape)
            self.annotated = np.zeros(self.starts.shape)
        self.HFO_type = HFO_type
        self.sample_freq = sample_freq
        self.feature_size = 0
        self.freq_range = freq_range
        self.time_range = time_range
        self.feature_size = feature_size
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

    def __str__(self):
        return "HFO_Feature: {} HFOs, {} artifacts, {} spkHFOs, {} eHFOs, {} real HFOs".format(
            self.num_HFO, self.num_artifact, self.num_spike, self.num_ehfo, self.num_real)
    @staticmethod
    def construct(channel_names, start_end, HFO_type="STE", sample_freq=2000, freq_range=[10, 500], time_range=[0, 1000], feature_size=224):
        '''
        Construct HFO_Feature object from detector output
        '''
        channel_names = np.concatenate([[channel_names[i]]*len(start_end[i]) for i in range(len(channel_names))])
        start_end = [se for se in start_end if len(se) > 0]
        start_end = np.concatenate(start_end) if start_end else np.empty((0, 2), dtype=int)

        # Filter out long event
        valid_indices = np.where((start_end[:, 1] - start_end[:, 0]) < sample_freq)[0] if start_end.size > 0 else np.array([])
        start_end = start_end[valid_indices] if valid_indices.size > 0 else np.empty((0, 2), dtype=int)
        channel_names = channel_names[valid_indices] if valid_indices.size > 0 else np.array([])
        return HFO_Feature(channel_names, start_end, np.array([]), HFO_type, sample_freq, freq_range, time_range, feature_size)
    
    def get_num_biomarker(self):
        return self.num_HFO
    
    def has_prediction(self):
        return self.artifact_predicted
    
    # def generate_psedo_label(self):
    #     self.artifact_predictions = np.ones(self.num_HFO)
    #     self.spike_predictions = np.zeros(self.num_HFO)
    #     self.artifact_predicted = True
    
    def doctor_annotation(self, annotation:str):
        # if annotation == "Artifact":
        #     self.artifact_annotations[self.index] = 0
        # elif annotation == "Spike":
        #     self.spike_annotations[self.index] = 1
        #     self.artifact_annotations[self.index] = 1
        # elif annotation == "Real":
        #     self.spike_annotations[self.index] = 0
        #     self.artifact_annotations[self.index] = 1
        if annotation == "Artifact":
            self.artifact_annotations[self.index] = 1
        elif annotation == "Pathological":
            self.pathological_annotations[self.index] = 1
        elif annotation == "Physiological":
            self.physiological_annotations[self.index] = 1
        self.annotated[self.index] = 1

    def get_next(self):
        if self.index >= self.num_HFO - 1:
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
    
    def _get_prediction(self, predictions):
        artifact_val = predictions.get("Artifact", 1)
        # -1 means undetected/unclassified, not artifact
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
        elif pathological_annotation == 1:
            return "Pathological"
        elif physiological_annotation == 1:
            return "Physiological"

    def get_current_info(self):
        # print("self.artifact_predicted:",self.artifact_predicted)
        channel_name = self.channel_names[self.index]
        start = self.starts[self.index]
        end = self.ends[self.index]

        prediction = self._get_prediction({'Artifact': self.artifact_predictions[self.index],
                                           'spkHFO': self.spike_predictions[self.index],
                                           'eHFO': self.ehfo_predictions[self.index]}) if self.artifact_predicted else None
        annotation = self._get_annotation(self.artifact_annotations[self.index],
                                          self.pathological_annotations[self.index],
                                          self.physiological_annotations[self.index]) if self.annotated[self.index] else None
        return {"channel_name": channel_name, "start_index": start, "end_index": end, "prediction": prediction, "annotation": annotation}

    def get_num_artifact(self):
        return self.num_artifact
    
    def get_num_spike(self):
        return self.num_spike

    def get_num_ehfo(self):
        return self.num_ehfo
    
    def get_num_real(self):
        return self.num_real
        
    def has_feature(self):
        return len(self.features) > 0

    def get_features(self):
        return self.features

    def get_raw_spectrums(self):
        return self.raw_spectrums

    def to_dict(self):
        channel_names = self.channel_names
        starts = self.starts
        ends = self.ends
        artifact_predictions = np.array(self.artifact_predictions)
        spike_predictions = np.array(self.spike_predictions)
        ehfo_predictions = np.array(self.ehfo_predictions)
        feature = self.features
        HFO_type = self.HFO_type
        sample_freq = self.sample_freq
        feature_size = self.feature_size
        freq_range = self.freq_range
        time_range = self.time_range
        return {"channel_names": channel_names, "starts": starts, "ends": ends,
                "artifact_predictions": artifact_predictions, "spike_predictions": spike_predictions,
                "ehfo_predictions": ehfo_predictions, "feature": feature, "HFO_type": HFO_type,
                "sample_freq": sample_freq, "feature_size": feature_size,
                "freq_range": freq_range, "time_range": time_range}
    
    @staticmethod
    def from_dict(data):
        '''
        construct HFO_Feature object from dictionary
        '''
        channel_names = data["channel_names"]
        starts = data["starts"]
        ends = data["ends"]
        artifact_predictions = data["artifact_predictions"]
        spike_predictions = data["spike_predictions"]
        ehfo_predictions = data['ehfo_predictions']
        feature = data["feature"]
        HFO_type = data["HFO_type"]
        sample_freq = data["sample_freq"]
        feature_size = data["feature_size"]
        freq_range = data["freq_range"]
        time_range = data["time_range"]
        biomarker_feature = HFO_Feature(channel_names, np.array([starts, ends]).T, feature, HFO_type, sample_freq, freq_range, time_range, feature_size)
        biomarker_feature.update_pred(artifact_predictions, spike_predictions, ehfo_predictions)

        return biomarker_feature

    def update_artifact_pred(self, artifact_predictions):
        self.artifact_predicted = True
        self.artifact_predictions = artifact_predictions
        self.num_artifact = np.sum(artifact_predictions <1)
        self.num_real = np.sum(artifact_predictions >0)
    
    def update_spike_pred(self, spike_predictions):
        self.spike_predicted = True
        self.spike_predictions = spike_predictions
        self.num_spike = np.sum(spike_predictions == 1)

    def update_ehfo_pred(self, ehfo_predictions):
        self.ehfo_predicted = True
        self.ehfo_predictions = ehfo_predictions
        self.num_ehfo = np.sum(ehfo_predictions == 1)
    
    def update_pred(self, artifact_predictions, spike_predictions, ehfo_predictions):
        artifact_predictions = np.array(artifact_predictions)
        # Only set artifact_predicted to True if there are actual predictions
        # (not all zeros = unclassified, not all -1 = undetected)
        if len(artifact_predictions) > 0:
            # Check if there are any actual predictions (values that are not 0 and not -1)
            # This means at least some items were classified
            has_actual_predictions = np.any((artifact_predictions != 0) & (artifact_predictions != -1))
            if has_actual_predictions:
                self.update_artifact_pred(artifact_predictions)
            else:
                # Items are unclassified (all zeros) or undetected (all -1)
                # Don't set artifact_predicted to True
                self.artifact_predictions = artifact_predictions
                self.artifact_predicted = False
        else:
            self.artifact_predictions = artifact_predictions
            self.artifact_predicted = False
        self.update_spike_pred(spike_predictions)
        self.update_ehfo_pred(ehfo_predictions)

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

    def get_biomarkers_for_channel(self, channel_name:str, min_start:int=None, max_end:int=None):
        channel_names = self.channel_names
        starts = self.starts
        ends = self.ends
        artifact_predictions = np.array(self.artifact_predictions)
        spike_predictions = np.array(self.spike_predictions)
        ehfo_predictions = np.array(self.ehfo_predictions)
        indexes = channel_names == channel_name
        if min_start is not None and max_end is not None:
            indexes = indexes & (starts >= min_start) & (ends <= max_end)
        starts = starts[indexes]
        ends = ends[indexes]
        try:
            artifact_predictions = artifact_predictions[indexes]
            spike_predictions = spike_predictions[indexes] == 1
            ehfo_predictions = ehfo_predictions[indexes] == 1
        except:
            artifact_predictions = []
            spike_predictions = []
            ehfo_predictions = []
        return starts, ends, artifact_predictions, spike_predictions, ehfo_predictions
    
    def get_annotation_text(self, index):
        channel_name = self.channel_names[index]
        if self.annotated[index] == 0:
            suffix = "Unannotated"
        elif self.artifact_annotations[index] == 1:
            suffix = "Artifact"
        # elif self.spike_annotations[index] == 1:
        #     suffix = "Spike"
        # else:
        #     suffix = "Real"
        elif self.pathological_annotations[index] == 1:
            suffix = "Pathological"
        elif self.physiological_annotations[index] == 1:
            suffix = "Physiological"
        else:
            import warnings
            warnings.warn("Annotation type not supported", category=UserWarning)

        return f" No.{index+1}: {channel_name} : {suffix}"

    def to_df(self):
        channel_names = self.channel_names
        starts = self.starts
        ends = self.ends
        artifact_predictions = np.array(self.artifact_predictions)
        spike_predictions = np.array(self.spike_predictions)
        ehfo_predictions = np.array(self.ehfo_predictions)
        artifact_annotations = np.array(self.artifact_annotations)
        # spike_annotations = np.array(self.spike_annotations)
        pathological_annotations = np.array(self.pathological_annotations)
        physiological_annotations = np.array(self.physiological_annotations)
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
        if len(ehfo_predictions) > 0:
            df["ehfo"] = ehfo_predictions
        df['annotated'] = annotated
        if len(artifact_annotations) > 0:
            df["artifact annotations"] = artifact_annotations
        # if len(spike_annotations) > 0:
        #     df["spike annotations"] = spike_annotations
        if len(pathological_annotations) > 0:
            df["pathological annotations"] = pathological_annotations
        if len(pathological_annotations) > 0:
            df["physiological annotations"] = physiological_annotations
        return df

    def export_csv(self, file_path):
        df = self.to_df()
        df.to_csv(file_path, index = False)
    
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
        df_out['annotated'] = 1 - (df_out["annotated"] > 0).astype(int)
        df_out["artifact annotations"] = (df_out["artifact annotations"] > 0).astype(int)
        df_out["pathological annotations"] = (df_out["pathological annotations"] > 0).astype(int)
        df_out["physiological annotations"] = (df_out["physiological annotations"] > 0).astype(int)
        df_channel = df_out.groupby("channel_names").agg({"starts": "count",
                                                          "artifact": "sum",
                                                          "spike": "sum",
                                                          "ehfo": "sum",
                                                          "annotated": "sum",
                                                          "artifact annotations": "sum",
                                                          "pathological annotations": "sum",
                                                          "physiological annotations": "sum"}).reset_index()
        df_channel.rename(columns={"starts": "Total Detection",
                                   "artifact": "HFO",
                                   "spike": "spkHFO",
                                   "ehfo": "eHFO",
                                   "annotated": "Unannotated",
                                   "artifact annotations": "HFO annotations"}, inplace=True)
        df.rename(columns={"artifact": "HFO", "spike": "spkHFO",
                           "annotated": "Annotated",
                           "artifact annotations": "HFO annotations"}, inplace=True)
        df['Annotated'] = df["Annotated"] > 0
        df['Annotated'] = df['Annotated'].replace({True: 'Yes', False: 'No'})
        with pd.ExcelWriter(file_path) as writer:
            df_channel.to_excel(writer, sheet_name="Channels", index=False)
            df.to_excel(writer, sheet_name="Events", index=False)