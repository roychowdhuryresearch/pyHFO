import numpy as np
import pandas as pd
class HFO_Feature():
    def __init__(self, channel_names, interval, features = [], HFO_type = "STE", sample_freq = 2000, freq_range = [10, 500], time_range = [0, 1000], feature_size = 224):
        self.channel_names = channel_names
        self.starts = interval[:, 0]
        self.ends = interval[:, 1]
        self.features = features
        self.artifact_predictions = []
        self.spike_predictions = []
        self.HFO_type = HFO_type
        self.sample_freq = sample_freq
        self.feature_size = 0
        self.freq_range = freq_range
        self.time_range = time_range
        self.feature_size = feature_size
        self.num_artifact = 0
        self.num_spike = 0
        self.num_HFO = len(self.starts)
        self.num_real = 0
        
    def __str__(self):
        return "HFO_Feature: {} HFOs, {} artifacts, {} spikes, {} real HFOs".format(self.num_HFO, self.num_artifact, self.num_spike, self.num_real)
    @staticmethod
    def construct(channel_names, start_end, HFO_type = "STE", sample_freq = 2000, freq_range = [10, 500], time_range = [0, 1000], feature_size = 224, prev_feature_to_join = None):
        '''
        Construct HFO_Feature object from detector output
        '''
        channel_key = {string: index for index, string in enumerate(channel_names)}
        channel_names = np.concatenate([[channel_names[i]]*len(start_end[i]) for i in range(len(channel_names))])
        start_end = [start_end[i] for i in range(len(start_end)) if len(start_end[i])>0]
        start_end = np.concatenate(start_end) if len(start_end) > 0 else np.array([])
        if prev_feature_to_join is not None:
            return HFO_Feature.joined_feature(channel_names, start_end, HFO_type, sample_freq, freq_range, time_range, feature_size, prev_feature_to_join, channel_key)
        return HFO_Feature(channel_names, start_end, np.array([]), HFO_type, sample_freq, freq_range, time_range, feature_size)
    
    def get_num_HFO(self):
        return self.num_HFO
    
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
        self.artifact_predictions = artifact_predictions
        self.num_artifact = np.sum(artifact_predictions <1)
        self.num_real = np.sum(artifact_predictions >0)
    
    def update_spike_pred(self, spike_predictions):
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

    def get_HFOs_for_channel(self, channel_name:str, min_start:int, max_end:int):
        channel_names = self.channel_names
        starts = self.starts
        ends = self.ends
        artifact_predictions = np.array(self.artifact_predictions)
        spike_predictions = np.array(self.spike_predictions)
        indexes = channel_names == channel_name
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

    def to_df(self):
        channel_names = self.channel_names
        starts = self.starts
        ends = self.ends
        artifact_predictions = np.array(self.artifact_predictions)
        spike_predictions = np.array(self.spike_predictions)
        df = pd.DataFrame()
        df["channel_names"] = channel_names
        df["starts"] = starts
        df["ends"] = ends
        if len(artifact_predictions) > 0:
            df["artifact"] = artifact_predictions
        if len(spike_predictions) > 0:
            df["spike"] = spike_predictions
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
        df_out["artifact"] = (df_out["artifact"] > 0).astype(int)
        df_out["spike"] = (df_out["spike"] > 0).astype(int)
        df_channel = df_out.groupby("channel_names").agg({"starts": "count","artifact": "sum", "spike":"sum"}).reset_index()
        df_channel.rename(columns={"start": "Total Detection", "artifact": "HFO", "spike": "spk-HFO"}, inplace=True)
        df.rename(columns={"artifact": "HFO", "spike": "spk-HFO"}, inplace=True)
        with pd.ExcelWriter(file_path) as writer:
            df_channel.to_excel(writer, sheet_name="Channels", index=False)
            df.to_excel(writer, sheet_name="Events", index=False)

    @staticmethod
    def joined_feature(channel_names, start_end, HFO_type, sample_freq, freq_range, time_range, feature_size, prev_feature, channel_key):
        if prev_feature is None: return
        if len(channel_names) == len(prev_feature.channel_names) == 0: return
        print(f'Joining the previous {prev_feature.HFO_type} detection')
        starts = start_end[:, 0]
        ends = start_end[:, 1]
        
        joined_ends = np.concatenate((ends, prev_feature.ends))
        joined_starts = np.concatenate((starts, prev_feature.starts))
        joined_channel_names = np.concatenate((channel_names, prev_feature.channel_names))
        # sort based on channel_name, then interval start
        joined_intervals = [list(inter) for inter in sorted(zip(joined_channel_names, joined_starts, joined_ends), key=lambda i : (channel_key[i[0]], i[1], i[2]))]
        merged_intervals = [joined_intervals[0]]
        for inter in joined_intervals:
            if merged_intervals[-1][0] == inter[0] and merged_intervals[-1][1] <= inter[1] <= merged_intervals[-1][-1]:
                merged_intervals[-1][-1] = max(merged_intervals[-1][-1], inter[-1])
            else:
                merged_intervals.append(inter)
        merged_intervals = np.array(merged_intervals)
        start_end = merged_intervals[:, 1:].astype(np.int64)
        channel_names = merged_intervals[:, 0]
        sample_freq = (sample_freq + prev_feature.sample_freq) // 2
        freq_range = [min(freq_range[0], prev_feature.freq_range[0]), max(freq_range[1], prev_feature.freq_range[1])]
        time_range = [min(time_range[0], prev_feature.time_range[0]), max(time_range[1], prev_feature.time_range[1])]
        feature_size = max(feature_size, prev_feature.feature_size)
        print(f'Joined {HFO_type} and {prev_feature.HFO_type} features')
        return HFO_Feature(channel_names, start_end, np.array([]), HFO_type, sample_freq, freq_range, time_range, feature_size)