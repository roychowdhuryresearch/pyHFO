import mne
import numpy as np
import scipy.signal as signal
# from models import ArtifactDetector, SpikeDetector
# import torch
from src.spindle_feature import SpindleFeature
from src.classifer import Classifier
from src.utils.utils_feature import *
from src.utils.utils_filter import construct_filter, filter_data
from src.utils.utils_detector import set_YASA_detector
from src.utils.utils_io import get_edf_info, read_eeg_data, dump_to_npz
from src.utils.utils_plotting import plot_feature

from src.param.param_detector import ParamDetector
from src.param.param_filter import ParamFilter
from src.param.param_classifier import ParamClassifier

import os
from p_tqdm import p_map
from pathlib import Path


class SpindleApp(object):
    def __init__(self):
        self.version = "1.0.0"
        self.biomarker_type = 'Spindle'
        self.n_jobs = 4
        ## eeg related
        self.eeg_data = None
        self.raw = None
        self.channel_names = None
        self.event_channel_names = None
        self.sample_freq = 0  # Hz
        self.edf_param = None

        ## filter related
        self.sos = None
        self.filter_data = None
        self.param_filter = None
        self.filtered = False

        # 60Hz filter related
        self.eeg_data_un60 = None
        self.filter_data_un60 = None
        self.eeg_data_60 = None
        self.filter_data_60 = None

        ## detector related
        self.param_detector = None
        self.detector = None
        self.detected = False

        ## feature related
        self.feature_param = None
        self.event_features = None

        ## classifier related
        self.param_classifier = None
        self.classifier = None
        self.classified = False
        self.Spindles = None

    def load_edf(self, file_path):
        print("Loading recording: " + file_path)
        if file_path.split(".")[-1] == "edf":
            self.raw = mne.io.read_raw_edf(file_path, verbose=0)
        # otherwise if its a brainvision file
        elif file_path.split(".")[-1] == "vhdr":
            # first check if the .eeg and .vmrk files also exist
            assert os.path.exists(
                file_path.replace(".vhdr", ".eeg")), "The .eeg file does not exist, cannot load the data"
            assert os.path.exists(
                file_path.replace(".vhdr", ".vmrk")), "The .vmrk file does not exist, cannot load the data"
            self.raw = mne.io.read_raw_brainvision(file_path, verbose=0)
        elif file_path.split(".")[-1] == "eeg":
            # first check if the .vhdr and .vmrk files also exist
            assert os.path.exists(
                file_path.replace(".eeg", ".vhdr")), "The .vhdr file does not exist, cannot load the data"
            assert os.path.exists(
                file_path.replace(".eeg", ".vmrk")), "The .vmrk file does not exist, cannot load the data"
            self.raw = mne.io.read_raw_brainvision(file_path.replace(".eeg", ".vhdr")
                                                   , verbose=0)
        elif file_path.split(".")[-1] == "vmrk":
            # first check if the .vhdr and .eeg files also exist
            assert os.path.exists(
                file_path.replace(".vmrk", ".vhdr")), "The .vhdr file does not exist, cannot load the data"
            assert os.path.exists(
                file_path.replace(".vmrk", ".eeg")), "The .eeg file does not exist, cannot load the data"
            self.raw = mne.io.read_raw_brainvision(file_path.replace(".vmrk", ".vhdr")
                                                   , verbose=0)
        else:
            raise ValueError("File type not supported")
        self.edf_param = get_edf_info(self.raw)
        self.sample_freq = int(self.edf_param['sfreq'])
        self.edf_param["edf_fn"] = file_path
        self.eeg_data, self.channel_names = read_eeg_data(self.raw)
        self.eeg_data_un60 = self.eeg_data.copy()
        self.eeg_data_60 = self.filter_60(self.eeg_data)
        # print("channel names: ", self.channel_names)
        # print("Loading COMPLETE!")

    def load_database(self):
        # @TODO; load database
        pass

    def get_edf_info(self):
        return self.edf_param

    def get_eeg_data(self, start: int = None, end: int = None, filtered: bool = False):
        data = self.eeg_data if not filtered else self.filter_data
        if start is None and end is None:
            return data, self.channel_names
        elif start is None:
            return data[:, :end], self.channel_names
        elif end is None:
            return data[:, start:], self.channel_names
        else:
            return data[:, start:end], self.channel_names

    def get_eeg_data_shape(self):
        return self.eeg_data.shape

    def get_sample_freq(self):
        return self.sample_freq

    def add_bipolar_channel(self, ch_1, ch_2):

        def bipolar(data, channels, ch1, ch2):
            return data[channels == ch1] - data[channels == ch2]

        bipolar_signal = bipolar(self.eeg_data, self.channel_names, ch_1, ch_2)
        bipolar_signalun60 = bipolar(self.eeg_data_un60, self.channel_names, ch_1, ch_2)
        bipolar_signal60 = bipolar(self.eeg_data_60, self.channel_names, ch_1, ch_2)

        if self.filtered == True:
            bipolar_filtered_60 = bipolar(self.filter_data_60, self.channel_names, ch_1, ch_2)
            bipolar_filtered_un60 = bipolar(self.filter_data_un60, self.channel_names, ch_1, ch_2)

        self.channel_names = np.concatenate([[f"{ch_1}#-#{ch_2}"], self.channel_names])

        # add filtered/unfiltered 60/un60 signals to different arrays
        self.eeg_data = np.concatenate([bipolar_signal, self.eeg_data])
        self.eeg_data_un60 = np.concatenate([bipolar_signalun60, self.eeg_data_un60])
        self.eeg_data_60 = np.concatenate([bipolar_signal60, self.eeg_data_60])
        if self.filtered == True:
            self.filtered_data_60 = np.concatenate([self.filter_data_60, bipolar_filtered_60])
            self.filtered_data_un60 = np.concatenate([self.filter_data_un60, bipolar_filtered_un60])
            self.filter_data = self.filtered_data_un60.copy()

    '''
        Filter API
    '''

    def set_filter_parameter(self, param_filter: ParamFilter):
        self.param_filter = param_filter
        self.param_filter.sample_freq = self.sample_freq
        sos = construct_filter(param_filter.fp, param_filter.fs, param_filter.rp, param_filter.rs, param_filter.space,
                               param_filter.sample_freq)
        # if any value in sos is nan, then raise error
        if np.isnan(sos).any():
            raise ValueError("filter parameter is invalid")
        self.sos = sos

    def filter_eeg_data(self, param_filter: ParamFilter = None):
        '''
        This is a function should be linked to the filter button
        '''

        if param_filter is not None:
            param_filter.sample_freq = self.sample_freq
            self.set_filter_parameter(param_filter)
        elif self.sos is None:
            raise ValueError("filter parameter is not set")

        self.filter_data = []
        param_list = [{"data": self.eeg_data_un60[i], "sos": self.sos} for i in range(len(self.eeg_data_un60))]
        # for i in range(len(param_list)):
        #     print("data shape:",param_list[i]["data"].shape, "sos shape:", param_list[i]["sos"].shape)
        ret = parallel_process(param_list, filter_data, n_jobs=self.n_jobs, use_kwargs=True, front_num=2)
        for r in ret:
            self.filter_data.append(r)
        self.filter_data = np.array(self.filter_data)
        self.filter_data_un60 = self.filter_data.copy()

        self.filter_data_60 = self.filter_60(self.filter_data)
        self.filtered = True

    def has_filtered_data(self):
        return self.filter_data is not None or len(self.filter_data) > 0

    def filter_60(self, data):
        """filter 60Hz noise"""
        filter_sos = signal.butter(5, [58, 62], 'bandstop', fs=self.sample_freq, output='sos')
        param_list = [{"data": data[i], "sos": filter_sos} for i in range(len(data))]
        ret = parallel_process(param_list, filter_data, n_jobs=self.n_jobs, use_kwargs=True, front_num=2)
        data_out = []
        for r in ret:
            data_out.append(r)
        return np.array(data_out)

    def set_filter_60(self):
        # self.eeg_data_un60 = self.eeg_data.copy()
        self.eeg_data = self.eeg_data_60.copy()
        if self.filtered:
            # self.filter_data_un60 = self.filter_data.copy()
            self.filter_data = self.filter_data_60.copy()

    def set_unfiltered_60(self):
        self.eeg_data = self.eeg_data_un60.copy()
        if self.filtered:
            self.filter_data = self.filter_data_un60.copy()

    '''
        Detector APIs
    '''

    def set_detector(self, param: ParamDetector):
        '''
        This is the function should be linked to the confirm button in the set detector window

        param should be a type of ParamDetector param/param_detector.py
        it should contain the following fields:
        detector_type: str, "STE", "MNI" or "HIL"
        detector_param: param_detector/ParamSTE or param_detector/ParamMNI

        '''

        self.param_detector = param
        # self.param_detector.detector_param.sample_freq = self.sample_freq
        # self.param_detector.detector_param.pass_band = int(self.param_filter.fp)
        # self.param_detector.detector_param.stop_band = int(self.param_filter.fs)
        # print("detector param: ", param.detector_param.to_dict())
        if param.detector_type.lower() == "yasa":
            self.detector = set_YASA_detector(param.detector_param)
        else:
            print('To be continued')


    def detect_biomarker(self, param_filter: ParamFilter = None, param_detector: ParamDetector = None):
        '''
        This the function should be linked to the detect button in the overview window,
        it can also be called with a param to set the detector, the detector will be reseted if the param is not None
        '''
        ## TODO: what is the detector param's filter is not the same as filter param
        # if param_filter is not None and not self.has_filtered_data():
        #     self.filter_eeg_data(param_filter)
        # if param_detector is not None:
        #     self.set_detector(param_detector)
        # if self.filter_data is None or len(self.filter_data) == 0:
        #     self.filter_eeg_data()
        # self.event_channel_names, self.HFOs = self.detector.detect_multi_channels(self.filter_data, self.channel_names,
        #                                                                           filtered=True)
        param_detector = self.detector['args']
        detector_yasa = self.detector['yasa']
        sp = detector_yasa.spindles_detect(self.eeg_data, sf=param_detector.sample_freq,
                                           ch_names=self.channel_names.tolist(), freq_sp=param_detector.freq_sp,
                                           freq_broad=param_detector.freq_broad, duration=param_detector.duration,
                                           min_distance=param_detector.min_distance, thresh={'corr': param_detector.corr,
                                                                                             'rel_pow': param_detector.rel_pow,
                                                                                             'rms': param_detector.rms})
        self.filter_data = sp._data_filt
        self.event_features = SpindleFeature.construct(sp, self.param_detector.detector_type, self.sample_freq)
        self.detected = True

    '''
        Feature APIs

    '''

    def generate_biomarker_features(self):
        '''
        Todo: feature generation parameter.
        '''
        freq_range = [10, self.edf_param['lowpass'] // 2]
        win_size = 224
        time_range = [0, 1000]  # 0~1000ms

        starts = self.event_features.starts
        ends = self.event_features.ends
        channel_names = self.event_features.channel_names
        hfo_waveforms = extract_waveforms(self.eeg_data, starts, ends, channel_names, self.channel_names,
                                          self.sample_freq, time_range)
        param_list = [{"start": starts[i], "end": ends[i], "data": hfo_waveforms[i], "channel_name": channel_names[i],
                       "sample_rate": self.sample_freq,
                       "win_size": win_size,
                       "ps_MinFreqHz": freq_range[0],
                       "ps_MaxFreqHz": freq_range[1],
                       "time_window_ms": (time_range[1] - time_range[0]) // 2,
                       } for i in range(len(starts))]
        ret = parallel_process(param_list, compute_biomarker_feature, n_jobs=self.n_jobs, use_kwargs=True, front_num=2)
        starts, ends, channel_names, time_frequncy_img, amplitude_coding_plot = np.zeros(len(ret)), np.zeros(
            len(ret)), np.empty(len(ret), dtype=object), np.zeros((len(ret), win_size, win_size)), np.zeros(
            (len(ret), win_size, win_size))
        for i in range(len(ret)):
            channel_names[i], starts[i], ends[i], time_frequncy_img[i], amplitude_coding_plot[i] = ret[i]
        interval = np.concatenate([starts[:, None], ends[:, None]], axis=1)
        feature = np.concatenate([time_frequncy_img[:, None, :, :], amplitude_coding_plot[:, None, :, :]], axis=1)
        self.event_features = SpindleFeature(channel_names, interval, feature, sample_freq=self.sample_freq,
                                          detector_type=self.param_detector.detector_type, feature_size=win_size,
                                          freq_range=freq_range, time_range=time_range)

    '''
        Classifier APIs

    '''

    def has_cuda(self):
        '''
        first check if cuda is available then set param
        if it returns true, then the the user can select device
        else the user can only select cpu
        '''
        # return torch.cuda.is_available()
        return False

    def get_classifier_param(self):
        ## todo: change it to database
        '''
        return the param_classifier This is the information should be shown in the ovewview window
        it also should be retrived when user set the classifier
        '''
        return self.param_classifier

    def set_classifier(self, param: ParamClassifier):
        '''
        This is the function should be linked to the confirm button in the set classifier window

        '''
        self.set_artifact_classifier(param)
        self.set_spike_classifier(param)

    def set_artifact_classifier(self, param: ParamClassifier):
        '''
        This is the function should be linked to the confirm button in the set artifact window

        '''

        self.param_classifier = param
        if self.classifier is None:
            self.classifier = Classifier(param)
        else:
            self.classifier.update_model_a(param)

    def set_spike_classifier(self, param: ParamClassifier):
        '''
        This is the function should be linked to the confirm button in the set spike window

        '''
        self.param_classifier = param
        self.classifier.update_model_s(param)

    def set_default_cpu_classifier(self):
        '''
        This is the function should be linked to the default cpu button in the set artifact window
        '''
        artifact_path = os.path.join(Path(os.path.dirname(__file__)).parent, "ckpt", "model_a.tar")
        spike_path = os.path.join(Path(os.path.dirname(__file__)).parent, "ckpt", "model_s.tar")
        self.param_classifier = ParamClassifier(artifact_path=artifact_path, spike_path=spike_path, use_spike=True,
                                                device="cpu", batch_size=32, model_type="default_cpu")
        self.classifier = Classifier(self.param_classifier)

    def set_default_gpu_classifier(self):
        '''
        This is the function should be linked to the default gpu button in the set artifact window
        '''
        artifact_path = os.path.join(Path(os.path.dirname(__file__)).parent, "ckpt", "model_a.tar")
        spike_path = os.path.join(Path(os.path.dirname(__file__)).parent, "ckpt", "model_s.tar")
        self.param_classifier = ParamClassifier(artifact_path=artifact_path, spike_path=spike_path, use_spike=True,
                                                device="cuda:0", batch_size=32, model_type="default_gpu")
        self.classifier = Classifier(self.param_classifier)

    def classify_artifacts(self, ignore_region=[1, 1], threshold=0.5):
        if not self.event_features.has_feature():
            self.generate_biomarker_features()
        ignore_region = np.array(ignore_region) * self.sample_freq
        ignore_region = np.array([ignore_region[0], len(self.eeg_data[0]) - ignore_region[1]])
        self.classifier.artifact_detection(self.event_features, ignore_region, threshold=threshold)
        self.classified = True

    def classify_spikes(self):
        if not self.event_features.has_feature():
            self.generate_biomarker_features()
        self.classifier.spike_detection(self.event_features)

    '''
        results APIs 
    '''

    def get_res_overview(self):
        '''
        return the overview of the results
        '''
        if not self.event_features.has_feature():
            self.generate_biomarker_features()
        return {
            "n_Spindle": self.event_features.num_spindle,
            "n_artifact": self.event_features.num_artifact,
            "n_real": self.event_features.num_real,
            "n_spike": self.event_features.num_spike
        }

    def export_report(self, path):
        if not self.event_features:
            return None
        self.event_features.export_csv(path)

    def export_excel(self, path):
        if not self.event_features:
            return None
        self.event_features.export_excel(path)

    def export_app(self, path):
        '''
        export all the data from app to a tar file
        '''
        checkpoint = {
            "n_jobs": self.n_jobs,
            "eeg_data": self.eeg_data,
            "edf_param": self.edf_param,
            "sample_freq": self.sample_freq,
            "channel_names": self.channel_names,
            "param_filter": self.param_filter.to_dict() if self.param_filter else None,
            "Spindles": self.Spindles,
            "param_detector": self.param_detector.to_dict() if self.param_detector else None,
            "Spindle_features": self.event_features.to_dict() if self.event_features else None,
            "param_classifier": self.param_classifier.to_dict() if self.param_classifier else None,
            "classified": self.classified,
            "filtered": self.filtered,
            "detected": self.detected,
            "artifact_predictions": np.array(self.event_features.artifact_predictions),
            "spike_predictions": np.array(self.event_features.spike_predictions),
            "artifact_annotations": np.array(self.event_features.artifact_annotations),
            "spike_annotations": np.array(self.event_features.spike_annotations),
            "annotated": np.array(self.event_features.annotated),
        }
        dump_to_npz(checkpoint, path)

    @staticmethod
    def import_app(path):
        '''
        import all the data from a tar file to app
        '''
        checkpoint = np.load(path, allow_pickle=True)
        app = SpindleApp()
        app.n_jobs = checkpoint["n_jobs"].item()
        app.eeg_data = checkpoint["eeg_data"]
        app.edf_param = checkpoint["edf_param"].item()
        app.sample_freq = checkpoint["sample_freq"]
        app.channel_names = checkpoint["channel_names"]
        app.classified = checkpoint["classified"].item()
        app.filtered = checkpoint["filtered"].item()
        app.detected = checkpoint["detected"].item()
        app.event_features.artifact_predictions = checkpoint["artifact_predictions"].item()
        app.event_features.spike_predictions = checkpoint["spike_predictions"].item()
        app.event_features.artifact_annotations = checkpoint["artifact_annotations"].item()
        app.event_features.spike_annotations = checkpoint["spike_annotations"].item()
        app.event_features.annotated = checkpoint["annotated"].item()
        if app.filtered:
            app.param_filter = ParamFilter.from_dict(checkpoint["param_filter"].item())
            app.filter_eeg_data(app.param_filter)
        if app.detected:
            # print("detected Spindles")
            app.Spindles = checkpoint["Spindles"]
            app.param_detector = ParamDetector.from_dict(checkpoint["param_detector"].item())
            # print("new Spindle features")
            app.event_features = SpindleFeature.from_dict(checkpoint["Spindle_features"].item())
            # print(app.event_features)
        if app.classified:
            app.param_classifier = ParamClassifier.from_dict(checkpoint["param_classifier"].item())
        return app

    def export_features(self, folder):
        def clean_folder(folder):
            import shutil
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder)

        def extract_data(data, data_filtered, start, end):
            data = np.squeeze(data)
            data_filtered = np.squeeze(data_filtered)
            if start < self.sample_freq // 2:
                plot_start, plot_end = 0, self.sample_freq
                hfo_start, hfo_end = start, min(end, self.sample_freq)
            elif end > len(data) - self.sample_freq // 2:
                plot_start, plot_end = len(data) - self.sample_freq, len(data)
                hfo_start, hfo_end = max(plot_start, start) - plot_start, min(plot_end, end) - plot_start
            else:
                plot_start, plot_end = (start + end) // 2 - self.sample_freq // 2, (
                            start + end) // 2 + self.sample_freq // 2
                hfo_start, hfo_end = max(plot_start, start) - plot_start, min(plot_end, end) - plot_start
            plot_start, plot_end, hfo_start, hfo_end = int(plot_start), int(plot_end), int(hfo_start), int(hfo_end)
            channel_data = data[plot_start:plot_end]
            channel_data_f = data_filtered[plot_start:plot_end]
            # print(hfo_start, hfo_end, start, end, plot_start, plot_end, channel_data.shape, channel_data_f.shape)
            return channel_data, channel_data_f, hfo_start, hfo_end

        def extract_waveform(data, data_filtered, starts, ends, channel_names, unique_channel_names):
            hfo_waveform_l, hfo_waveform_f_l, hfo_start_l, hfo_end_l = np.zeros((len(starts), 2000)), np.zeros(
                (len(starts), 2000)), [], []
            for i in tqdm(range(len(starts))):
                channel_name = channel_names[i]
                start = starts[i]
                end = ends[i]
                channel_index = np.where(unique_channel_names == channel_name)[0]
                hfo_waveform, hfo_waveform_f, hfo_start, hfo_end = extract_data(data[channel_index],
                                                                                data_filtered[channel_index], start,
                                                                                end)
                hfo_waveform_l[i] = hfo_waveform
                hfo_waveform_f_l[i] = hfo_waveform_f
                hfo_start_l.append(hfo_start)
                hfo_end_l.append(hfo_end)
            return hfo_waveform_l, hfo_waveform_f_l, np.array(hfo_start_l), np.array(hfo_end_l)

        if not self.event_features:
            return None
        os.makedirs(folder, exist_ok=True)
        artifact_folder = os.path.join(folder, "artifact")
        spike_folder = os.path.join(folder, "spike")
        non_spike_folder = os.path.join(folder, "non_spike")
        clean_folder(artifact_folder)
        clean_folder(spike_folder)
        clean_folder(non_spike_folder)
        starts = self.event_features.starts
        ends = self.event_features.ends
        feature = self.event_features.features
        channel_names = self.event_features.channel_names
        spike_predictions = self.event_features.spike_predictions
        index_s = np.where(spike_predictions == 1)[0]
        start_s, end_s, feature_s, channel_names_s = starts[index_s], ends[index_s], feature[index_s], channel_names[
            index_s]
        index_a = np.where(spike_predictions == -1)[0]
        start_a, end_a, feature_a, channel_names_a = starts[index_a], ends[index_a], feature[index_a], channel_names[
            index_a]
        index_r = np.where(spike_predictions == 0)[0]
        start_r, end_r, feature_r, channel_names_r = starts[index_r], ends[index_r], feature[index_r], channel_names[
            index_r]
        # print("plotting Spindle with spike")
        waveform_s, waveform_f_s, hfo_start_s, hfo_end_s = extract_waveform(self.eeg_data, self.filter_data, start_s,
                                                                            end_s, channel_names_s, self.channel_names)
        param_list = [{"folder": spike_folder, "start": start_s[i], "end": end_s[i], "feature": feature_s[i],
                       "channel_name": channel_names_s[i], "data": waveform_s[i], "data_filtered": waveform_f_s[i],
                       "hfo_start": hfo_start_s[i], "hfo_end": hfo_end_s[i]} for i in range(len(start_s))]
        ret = parallel_process(param_list, plot_feature, self.n_jobs, use_kwargs=True, front_num=3)
        waveform_a, waveform_f_a, hfo_start_a, hfo_end_a = extract_waveform(self.eeg_data, self.filter_data, start_a,
                                                                            end_a, channel_names_a, self.channel_names)
        param_list = [{"folder": artifact_folder, "start": start_a[i], "end": end_a[i], "feature": feature_a[i],
                       "channel_name": channel_names_a[i], "data": waveform_a[i], "data_filtered": waveform_f_a[i],
                       "hfo_start": hfo_start_a[i], "hfo_end": hfo_end_a[i]} for i in range(len(start_a))]
        ret = parallel_process(param_list, plot_feature, self.n_jobs, use_kwargs=True, front_num=3)
        waveform_r, waveform_f_r, hfo_start_r, hfo_end_r = extract_waveform(self.eeg_data, self.filter_data, start_r,
                                                                            end_r, channel_names_r, self.channel_names)
        param_list = [{"folder": non_spike_folder, "start": start_r[i], "end": end_r[i], "feature": feature_r[i],
                       "channel_name": channel_names_r[i], "data": waveform_r[i], "data_filtered": waveform_f_r[i],
                       "hfo_start": hfo_start_r[i], "hfo_end": hfo_end_r[i]} for i in range(len(start_r))]
        ret = parallel_process(param_list, plot_feature, self.n_jobs, use_kwargs=True, front_num=3)
