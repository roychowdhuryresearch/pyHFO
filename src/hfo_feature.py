import numpy as np
import pandas as pd
class HFO_Feature():
    def __init__(self, channel_names, start_end, features = [], HFO_type = "STE", sample_freq = 2000, freq_range = [10, 500], time_range = [0, 1000], feature_size = 224, num_HFO = 0, num_spindle = 0):
        self.channel_names = channel_names
        
        try:
            self.starts = start_end[:, 0]
        except:
            self.starts = []
        try:
            self.ends = start_end[:, 1]
        except:
            self.ends = []

        self.start_end = start_end
        self.features = features
        self.artifact_predictions = np.zeros(self.starts.shape)
        self.spike_predictions = []
        self.artifact_annotations = np.zeros(self.starts.shape)
        self.spike_annotations = np.zeros(self.starts.shape)
        self.annotated = np.zeros(self.starts.shape)
        self.HFO_type = HFO_type
        self.sample_freq = sample_freq
        self.feature_size = 0
        self.freq_range = freq_range
        self.time_range = time_range
        self.feature_size = feature_size
        self.num_artifact = 0
        self.num_spike = 0
        self.num_HFO = num_HFO
        self.num_spindle = num_spindle
        self.num_real = 0
        self.index = 0
        self.artifact_predicted = False
        self.spike_predicted = False

    
    def __str__(self):
        return "HFO_Feature: {} HFOs, {} artifacts, {} spikes, {} real HFOs".format(self.num_HFO, self.num_artifact, self.num_spike, self.num_real)
    @staticmethod
    def construct(channel_names, start_end, HFO_type = "STE", sample_freq = 2000, freq_range = [10, 500], time_range = [0, 1000], feature_size = 224):
        '''
        Construct HFO_Feature object from detector output
        '''

        if HFO_type.lower() == "spindle":
            return HFO_Feature(channel_names, start_end, np.array([]), HFO_type, sample_freq, freq_range, time_range, feature_size, num_HFO = 0, num_spindle = len(start_end))
        if HFO_type.lower() == "spike":
            return HFO_Feature(channel_names, start_end, np.array([]), HFO_type, sample_freq, freq_range, time_range, feature_size, num_HFO = 0, num_spindle = 0)
        
        # if output is from STE or MNI detector, need operations to flatten the result
        channel_names = np.concatenate([[channel_names[i]]*len(start_end[i]) for i in range(len(channel_names))])
        start_end = [start_end[i] for i in range(len(start_end)) if len(start_end[i])>0]
        start_end = np.concatenate(start_end) if len(start_end) > 0 else np.array([])
        return HFO_Feature(channel_names, start_end, np.array([]), HFO_type, sample_freq, freq_range, time_range, feature_size,  num_HFO = len(start_end), num_spindle = 0)
    
    def get_num_HFO(self):
        return self.num_HFO
    
    def has_prediction(self):
        return self.artifact_predicted
    
    # def generate_psedo_label(self):
    #     self.artifact_predictions = np.ones(self.num_HFO)
    #     self.spike_predictions = np.zeros(self.num_HFO)
    #     self.artifact_predicted = True
    
    def doctor_annotation(self, annotation:str):
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
    
    def _get_prediction(self, artifact_prediction, spike_prediction):
        if artifact_prediction < 1:
            return "Artifact"
        elif spike_prediction == 1:
            return "Spike"
        else:
            return "HFO"

    def get_current_info(self):
        print("self.artifact_predicted:",self.artifact_predicted)
        channel_name = self.channel_names[self.index]
        start = self.starts[self.index]
        end = self.ends[self.index]
        prediction = self._get_prediction(self.artifact_predictions[self.index], self.spike_predictions[self.index]) if self.artifact_predicted else None
        annotation = self._get_prediction(self.artifact_annotations[self.index], self.spike_annotations[self.index]) if self.annotated[self.index] else None
        return {"channel_name": channel_name, "start_index": start, "end_index": end, "prediction": prediction, "annotation": annotation}

                    
    def get_num_artifact(self):
        return self.num_artifact
    
    def get_num_spike(self):
        return self.num_spike
    
    def get_num_real(self):
        return self.num_real
    
    def get_num_Spindle(self):
        return self.num_spindle
        
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
        HFO_type = self.HFO_type
        sample_freq = self.sample_freq
        feature_size = self.feature_size
        freq_range = self.freq_range
        time_range = self.time_range
        return {"channel_names": channel_names, "starts": starts, "ends": ends, "artifact_predictions": artifact_predictions, "spike_predictions": spike_predictions, "feature": feature, "HFO_type": HFO_type, "sample_freq": sample_freq, "feature_size": feature_size, "freq_range": freq_range, "time_range": time_range}
    
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
        feature = data["feature"]
        HFO_type = data["HFO_type"]
        sample_freq = data["sample_freq"]
        feature_size = data["feature_size"]
        freq_range = data["freq_range"]
        time_range = data["time_range"]
        hfo_feature = HFO_Feature(channel_names, np.array([starts, ends]).T, feature, HFO_type, sample_freq, freq_range, time_range, feature_size)
        hfo_feature.update_pred(artifact_predictions, spike_predictions)

        return hfo_feature

    def update_artifact_pred(self, artifact_predictions):
        self.artifact_predicted = True
        self.artifact_predictions = artifact_predictions
        self.num_artifact = np.sum(artifact_predictions <1)
        self.num_real = np.sum(artifact_predictions >0)
    
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

    def get_HFOs_for_channel(self, channel_name:str, min_start:int=None, max_end:int=None):
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
        return f" No.{index+1}: {channel_name} : {suffix}"

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
        df.to_csv(file_path, index = False)
    
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
                                                          "artifact annotations": "sum", "spike annotations": "sum"}).reset_index()
        df_channel.rename(columns={"starts": "Total Detection",
                                   "artifact": "HFO", "spike": "spk-HFO",
                                   "annotated": "Unannotated",
                                   "artifact annotations": "HFO annotations", "spike annotations": "spk-HFO annotations"}, inplace=True)
        df.rename(columns={"artifact": "HFO", "spike": "spk-HFO",
                           "annotated": "Annotated",
                           "artifact annotations": "HFO annotations", "spike annotations": "spk-HFO annotations"}, inplace=True)
        df['Annotated'] = df["Annotated"] > 0
        df['Annotated'] = df['Annotated'].replace({True: 'Yes', False: 'No'})
        with pd.ExcelWriter(file_path) as writer:
            df_channel.to_excel(writer, sheet_name="Channels", index=False)
            df.to_excel(writer, sheet_name="Events", index=False)

    @staticmethod
    def join_features(feature1, feature2, channel_names):
        if feature1 is None and feature2 is not None: return (feature2.start_end, feature2)
        elif feature2 is None: return (feature1.start_end, feature1)
        if len(feature1.channel_names) == len(feature2.channel_names) == 0: 
            return ([], [])
        print(f'Joining {feature1.HFO_type} and {feature2.HFO_type} detection')
        
        channel_key = {string: index for index, string in enumerate(channel_names)}
        joined_ends = np.concatenate((feature1.ends, feature2.ends))
        joined_starts = np.concatenate((feature1.starts, feature2.starts))
        joined_channel_names = np.concatenate((feature1.channel_names, feature2.channel_names))
        # sort based on channel_name, then interval start, with channel key as the order of channels
        joined_intervals = [list(inter) for inter in sorted(zip(joined_channel_names, joined_starts, joined_ends), key=lambda i : (channel_key[i[0]], i[1], i[2]))]

        if len(joined_intervals) == 0:
            merged_intervals = []
        else:
            # put the first interval in the merged list
            merged_intervals = [joined_intervals[0]]
            for inter in joined_intervals:
                # if in the same channel and overlaps
                if merged_intervals[-1][0] == inter[0] and merged_intervals[-1][1] <= inter[1] <= merged_intervals[-1][-1]:
                    merged_intervals[-1][-1] = max(merged_intervals[-1][-1], inter[-1])
                else:
                    merged_intervals.append(inter)

        merged_intervals = np.array(merged_intervals)
        start_end = merged_intervals[:, 1:].astype(np.int64)
        channel_names = merged_intervals[:, 0]
        sample_freq = (feature1.sample_freq + feature2.sample_freq) // 2
        freq_range = [min(feature1.freq_range[0], feature2.freq_range[0]), max(feature1.freq_range[1], feature2.freq_range[1])]
        time_range = [min(feature1.time_range[0], feature2.time_range[0]), max(feature1.time_range[1], feature2.time_range[1])]
        feature_size = max(feature1.feature_size, feature2.feature_size)
        total_HFO = feature1.num_HFO + feature2.num_HFO
        total_spindle = feature1.num_spindle + feature2.num_spindle
        print(f'Joined {feature1.HFO_type} and {feature2.HFO_type} features')
        joined_feature = HFO_Feature(channel_names, start_end, np.array([]), "Joined", sample_freq, freq_range, time_range, feature_size, total_HFO, total_spindle)
        return (joined_feature.start_end, joined_feature)