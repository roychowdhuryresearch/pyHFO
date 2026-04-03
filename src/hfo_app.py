import mne
import numpy as np
import pandas as pd
import scipy.signal as signal
# from models import ArtifactDetector, SpikeDetector
# import torch
from src.hfo_feature import HFO_Feature
from src.utils.utils_feature import *
from src.utils.utils_filter import construct_filter, filter_data
from src.utils.utils_detector import set_STE_detector, set_MNI_detector, set_HIL_detector
from src.utils.utils_io import get_edf_info, read_eeg_data, dump_to_npz, load_mne_raw
from src.utils.utils_plotting import plot_feature
from src.utils.app_state import build_base_checkpoint, checkpoint_array, checkpoint_get
from src.utils.analysis_session import AnalysisSession, DetectionRun
from src.utils.session_store import load_session_checkpoint, save_session_checkpoint

from src.param.param_detector import ParamDetector
from src.param.param_filter import ParamFilter
from src.param.param_classifier import ParamClassifier

import os
from p_tqdm import p_map
from pathlib import Path

class HFO_App(object):
    def __init__(self):
        self.version = "1.0.0"
        self.biomarker_type = 'HFO'
        self.n_jobs = 4
        ## eeg related
        self.eeg_data = None
        self.raw = None
        self.channel_names = None
        self.event_channel_names = None
        self.sample_freq = 0 # Hz
        self.edf_param = None

        ## filter related
        self.sos = None
        self.filter_data = None
        self.param_filter = None
        self.filtered = False

        #60Hz filter related
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
        self.HFOs = None
        self.analysis_session = AnalysisSession(self.biomarker_type)

    def set_n_jobs(self, n_jobs):
        self.n_jobs = int(n_jobs)
        if self.param_detector is not None:
            self.param_detector.detector_param.n_jobs = self.n_jobs
            if self.param_filter is not None:
                self.set_detector(self.param_detector)

    def _ensure_classifier(self, param: ParamClassifier | None = None):
        if self.classifier is None:
            if param is None:
                param = self.param_classifier
            if param is None:
                raise ValueError("Classifier parameters are not set.")
            from src.classifer import Classifier
            self.classifier = Classifier(param)
        return self.classifier

    def load_raw(self, raw, file_path="<memory>"):
        self.raw = raw
        self.edf_param = get_edf_info(self.raw)
        self.sample_freq = int(self.edf_param['sfreq'])
        self.edf_param["edf_fn"] = file_path
        self.eeg_data, self.channel_names = read_eeg_data(self.raw)
        self.eeg_data_un60 = self.eeg_data
        self.eeg_data_60 = None
        self.analysis_session = AnalysisSession(self.biomarker_type)
        return self



    def load_edf(self, file_path):
        print("Loading recording: " + file_path)
        self.load_raw(load_mne_raw(file_path), file_path=file_path)
        # print("channel names: ", self.channel_names)
        # print("Loading COMPLETE!")
    
    def load_database(self):
        #@TODO; load database
        pass

    def get_edf_info(self):
        return self.edf_param

    def get_eeg_data(self,start:int=None, end:int=None, filtered:bool=False):
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

        def bipolar(data,channels,ch1,ch2):
            return data[channels==ch1]-data[channels==ch2]

        bipolar_signal = bipolar(self.eeg_data,self.channel_names,ch_1,ch_2)
        bipolar_signalun60 = bipolar(self.eeg_data_un60,self.channel_names,ch_1,ch_2)
        bipolar_signal60 = bipolar(self.ensure_eeg_data_60(), self.channel_names, ch_1, ch_2)

        if self.filtered == True:
            bipolar_filtered_60 = bipolar(self.ensure_filter_data_60(), self.channel_names, ch_1, ch_2)
            bipolar_filtered_un60 = bipolar(self.filter_data_un60, self.channel_names, ch_1, ch_2)

        self.channel_names = np.concatenate([[f"{ch_1}#-#{ch_2}"],self.channel_names])

        #add filtered/unfiltered 60/un60 signals to different arrays 
        self.eeg_data = np.concatenate([bipolar_signal,self.eeg_data])
        self.eeg_data_un60 = np.concatenate([bipolar_signalun60,self.eeg_data_un60])
        self.eeg_data_60 = np.concatenate([bipolar_signal60, self.ensure_eeg_data_60()])
        if self.filtered == True: 
            self.filter_data_60 = np.concatenate([self.ensure_filter_data_60(), bipolar_filtered_60])
            self.filter_data_un60 = np.concatenate([self.filter_data_un60, bipolar_filtered_un60])
            self.filter_data = self.filter_data_un60.copy()

    '''
        Filter API
    '''

    def set_filter_parameter(self, param_filter:ParamFilter):
        self.param_filter = param_filter
        self.param_filter.sample_freq = self.sample_freq
        sos = construct_filter(param_filter.fp, param_filter.fs, param_filter.rp, param_filter.rs, param_filter.space, param_filter.sample_freq)
        #if any value in sos is nan, then raise error
        if np.isnan(sos).any():
            raise ValueError("filter parameter is invalid")
        self.sos = sos


    def filter_eeg_data(self, param_filter:ParamFilter=None):
        '''
        This is a function should be linked to the filter button
        '''

        if param_filter is not None:
            param_filter.sample_freq = self.sample_freq
            self.set_filter_parameter(param_filter)
        elif self.sos is None:
            raise ValueError("filter parameter is not set")

        self.filter_data = []
        param_list = [{"data":self.eeg_data_un60[i], "sos":self.sos} for i in range(len(self.eeg_data_un60))]
        # for i in range(len(param_list)):
        #     print("data shape:",param_list[i]["data"].shape, "sos shape:", param_list[i]["sos"].shape)
        ret = parallel_process(param_list, filter_data, n_jobs=self.n_jobs, use_kwargs=True, front_num=2)
        for r in ret:
            if isinstance(r, Exception):
                raise r
            self.filter_data.append(r)
        self.filter_data = np.array(self.filter_data)
        self.filter_data_un60 = self.filter_data.copy()
        self.filter_data_60 = None
        self.filtered = True

    def has_filtered_data(self):
        return self.filter_data is not None and len(self.filter_data) > 0
    
    def compute_60hz_filtered_data(self, data):
        """filter 60Hz noise"""
        filter_sos = signal.butter(5, [58, 62], 'bandstop', fs=self.sample_freq, output='sos')
        param_list = [{"data": data[i], "sos": filter_sos} for i in range(len(data))]
        ret = parallel_process(param_list, filter_data, n_jobs=self.n_jobs, use_kwargs=True, front_num=2)
        data_out = []
        for r in ret:
            if isinstance(r, Exception):
                raise r
            data_out.append(r)
        return np.array(data_out)

    def ensure_eeg_data_60(self):
        if self.eeg_data_60 is None and self.eeg_data_un60 is not None:
            self.eeg_data_60 = self.compute_60hz_filtered_data(self.eeg_data_un60)
        return self.eeg_data_60

    def ensure_filter_data_60(self):
        if self.filter_data_60 is None and self.filter_data_un60 is not None:
            self.filter_data_60 = self.compute_60hz_filtered_data(self.filter_data_un60)
        return self.filter_data_60
    
    def set_filter_60(self):
        # self.eeg_data_un60 = self.eeg_data.copy()
        self.eeg_data = self.ensure_eeg_data_60().copy()
        if self.filtered:
            # self.filter_data_un60 = self.filter_data.copy()
            self.filter_data = self.ensure_filter_data_60().copy()

    def set_unfiltered_60(self):
        self.eeg_data = self.eeg_data_un60.copy()
        if self.filtered:
            self.filter_data = self.filter_data_un60.copy()
        



    '''
        Detector APIs
    '''

    def set_detector(self, param:ParamDetector):
        '''
        This is the function should be linked to the confirm button in the set detector window

        param should be a type of ParamDetector param/param_detector.py
        it should contain the following fields:
        detector_type: str, "STE", "MNI" or "HIL"
        detector_param: param_detector/ParamSTE or param_detector/ParamMNI
        
        '''

        self.param_detector = param
        self.param_detector.detector_param.sample_freq = self.sample_freq
        self.param_detector.detector_param.pass_band = int(self.param_filter.fp)
        self.param_detector.detector_param.stop_band = int(self.param_filter.fs)
        #print("detector param: ", param.detector_param.to_dict())
        if param.detector_type.lower() == "ste":
            self.detector = set_STE_detector(param.detector_param)
        elif param.detector_type.lower() == "mni":
            self.detector = set_MNI_detector(param.detector_param)
        elif param.detector_type.lower() == "hil":
            self.detector = set_HIL_detector(param.detector_param)          
    def detect_biomarker(self, param_filter:ParamFilter =None, param_detector:ParamDetector=None):
        '''
        This the function should be linked to the detect button in the overview window, 
        it can also be called with a param to set the detector, the detector will be reseted if the param is not None
        '''
        ## TODO: what is the detector param's filter is not the same as filter param\
        if param_filter is not None and not self.has_filtered_data():
            self.filter_eeg_data(param_filter)
        if param_detector is not None:
            self.set_detector(param_detector)
        if self.filter_data is None or len(self.filter_data) == 0:
            self.filter_eeg_data()
        self.event_channel_names, self.HFOs = self.detector.detect_multi_channels(self.filter_data, self.channel_names, filtered=True)
        self.event_features = HFO_Feature.construct(self.event_channel_names, self.HFOs, self.param_detector.detector_type, self.sample_freq)
        self.detected = True
        self.classified = False
        self.capture_current_run()

    '''
        Feature APIs
    
    '''

    def generate_HFO_features(self):
        '''
        Todo: feature generation parameter. 
        '''
        freq_range = [10, self.edf_param['lowpass']//2]
        win_size = 224
        time_range = [0, 1000] # 0~1000ms

        starts = self.event_features.starts
        ends = self.event_features.ends
        channel_names = self.event_features.channel_names
        biomarker_waveforms = extract_waveforms(self.eeg_data, starts, ends, channel_names, self.channel_names, self.sample_freq, time_range)
        param_list = [{"start":starts[i], "end":ends[i], "data":biomarker_waveforms[i], "channel_name":channel_names[i],
                       "sample_rate": self.sample_freq,
                       "win_size": win_size,
                       "ps_MinFreqHz": freq_range[0],
                        "ps_MaxFreqHz": freq_range[1],
                        "time_window_ms" : (time_range[1] - time_range[0])//2,
                       } for i in range(len(starts))]
        ret = parallel_process(param_list, compute_biomarker_feature, n_jobs=self.n_jobs, use_kwargs=True, front_num=2)
        starts, ends, channel_names, time_frequncy_img, amplitude_coding_plot, raw_spectrums = np.zeros(len(ret)), np.zeros(len(ret)), np.empty(len(ret), dtype= object), np.zeros((len(ret), win_size,win_size)), np.zeros((len(ret), win_size, win_size)), []
        for i in range(len(ret)):
            channel_names[i], starts[i], ends[i], time_frequncy_img[i], amplitude_coding_plot[i],  raw_spectrum= ret[i]
            raw_spectrums.append(raw_spectrum)
        raw_spectrums = np.array(raw_spectrums)
        interval = np.concatenate([starts[:, None], ends[:, None]], axis=1)
        feature = np.concatenate([time_frequncy_img[:, None, :, :], amplitude_coding_plot[:, None, :, :]], axis=1)
        self.event_features = HFO_Feature(channel_names, interval, feature, sample_freq = self.sample_freq, HFO_type=self.param_detector.detector_type, feature_size=win_size, freq_range=freq_range, time_range=time_range, raw_spectrums=raw_spectrums)


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

    def set_classifier(self, param:ParamClassifier):
        '''
        This is the function should be linked to the confirm button in the set classifier window
        
        '''
        self.set_artifact_classifier(param)
        self.set_spike_classifier(param)
        self.set_ehfo_classifier(param)


    def set_artifact_classifier(self, param:ParamClassifier):
        '''
        This is the function should be linked to the confirm button in the set artifact window
        
        '''
 
        self.param_classifier = param
        classifier = self._ensure_classifier(param)

        if param.artifact_card:
            classifier.update_model_artifact(param)
        elif param.artifact_path:
            classifier.update_model_a(param)

    def set_spike_classifier(self, param:ParamClassifier):
        '''
        This is the function should be linked to the confirm button in the set spike window
        
        '''
        self.param_classifier = param
        classifier = self._ensure_classifier(param)

        if not param.use_spike:
            return

        if param.spike_card:
            classifier.update_model_spkhfo(param)
        elif param.spike_path:
            classifier.update_model_s(param)

    def set_ehfo_classifier(self, param: ParamClassifier):
        '''
        This is the function should be linked to the confirm button in the set spike window

        '''
        self.param_classifier = param
        classifier = self._ensure_classifier(param)

        if not param.use_ehfo:
            return

        if param.ehfo_card:
            classifier.update_model_ehfo(param)
        elif param.ehfo_path:
            classifier.update_model_e(param)

    def set_default_cpu_classifier(self):
        '''
        This is the function should be linked to the default cpu button in the set artifact window
        '''
        artifact_path = os.path.join(Path(os.path.dirname(__file__)).parent, "ckpt", "model_a.tar")
        spike_path = os.path.join(Path(os.path.dirname(__file__)).parent, "ckpt", "model_s.tar")
        ehfo_path = os.path.join(Path(os.path.dirname(__file__)).parent, "ckpt", "model_e.tar")
        artifact_card = 'roychowdhuryresearch/HFO-artifact'
        spike_card = 'roychowdhuryresearch/HFO-spkHFO'
        ehfo_card = 'roychowdhuryresearch/HFO-eHFO'
        self.param_classifier = ParamClassifier(artifact_path=artifact_path, spike_path=spike_path, ehfo_path=ehfo_path,
                                                artifact_card=artifact_card, spike_card=spike_card, ehfo_card=ehfo_card,
                                                use_spike=True, use_ehfo=True,
                                                device="cpu", batch_size=32, model_type="default_cpu")
        self.classifier = None

    def set_default_gpu_classifier(self):
        '''
        This is the function should be linked to the default gpu button in the set artifact window
        '''
        artifact_path = os.path.join(Path(os.path.dirname(__file__)).parent, "ckpt", "model_a.tar")
        spike_path = os.path.join(Path(os.path.dirname(__file__)).parent, "ckpt", "model_s.tar")
        ehfo_path = os.path.join(Path(os.path.dirname(__file__)).parent, "ckpt", "model_e.tar")
        artifact_card = 'roychowdhuryresearch/HFO-artifact'
        spike_card = 'roychowdhuryresearch/HFO-spkHFO'
        ehfo_card = 'roychowdhuryresearch/HFO-eHFO'
        self.param_classifier = ParamClassifier(artifact_path=artifact_path, spike_path=spike_path, ehfo_path=ehfo_path,
                                                artifact_card=artifact_card, spike_card=spike_card, ehfo_card=ehfo_card,
                                                use_spike=True, use_ehfo=True,
                                                device="cuda:0", batch_size=32, model_type="default_gpu")
        self.classifier = None


    def classify_artifacts(self, ignore_region = [1, 1], threshold=0.5):
        if not self.event_features.has_feature():
            self.generate_HFO_features()
        self._ensure_classifier()
        ignore_region = np.array(ignore_region) * self.sample_freq
        ignore_region = np.array([ignore_region[0], len(self.eeg_data[0]) - ignore_region[1]])
        self.classifier.artifact_detection(self.event_features, ignore_region, threshold=threshold)
        self.classified = True
        self.sync_active_run()
    
    def classify_spikes(self):
        if not self.event_features.has_feature():
            self.generate_HFO_features()
        self._ensure_classifier()
        self.classifier.spike_detection(self.event_features)
        self.sync_active_run()

    def classify_ehfos(self):
        if not self.event_features.has_feature():
            self.generate_HFO_features()
        self._ensure_classifier()
        self.classifier.ehfo_detection(self.event_features)
        self.sync_active_run()

    '''
        results APIs 
    '''

    def get_res_overview(self):
        '''
        return the overview of the results
        '''
        if not self.event_features.has_feature():
            self.generate_HFO_features()
        return {
            "n_HFO": self.event_features.num_HFO,
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
        save_session_checkpoint(path, self.to_checkpoint())
    
    @staticmethod 
    def import_app(path):
        '''
        import all the data from a tar file to app
        '''
        checkpoint = load_session_checkpoint(path)
        app = HFO_App()
        app.load_checkpoint(checkpoint)
        return app

    def capture_current_run(self):
        if self.event_features is None:
            return None
        detector_name = self.param_detector.detector_type if self.param_detector else self.biomarker_type
        run = DetectionRun.create(
            biomarker_type=self.biomarker_type,
            detector_name=detector_name,
            selected_channels=self.event_channel_names if self.event_channel_names is not None else self.channel_names,
            param_filter=self.param_filter,
            param_detector=self.param_detector,
            param_classifier=self.param_classifier,
            event_features=self.event_features,
            detector_output=self.HFOs,
            classified=self.classified,
        )
        self.analysis_session.add_run(run)
        return run

    def sync_active_run(self):
        run = self.analysis_session.get_active_run()
        if run is None or self.event_features is None:
            return
        run.param_filter = self.param_filter
        run.param_detector = self.param_detector
        run.param_classifier = self.param_classifier
        run.event_features = self.event_features
        run.detector_output = self.HFOs
        run.classified = self.classified
        run.selected_channels = list(np.array(self.event_channel_names if self.event_channel_names is not None else self.channel_names).tolist())
        run.refresh_summary()

    def activate_run(self, run_id):
        self.analysis_session.activate_run(run_id)
        run = self.analysis_session.get_active_run()
        if run is None:
            return
        self.param_filter = run.param_filter
        self.param_detector = run.param_detector
        self.param_classifier = run.param_classifier
        self.event_features = run.event_features
        self.HFOs = run.detector_output
        self.event_channel_names = np.array(run.selected_channels) if run.selected_channels else self.channel_names
        self.detected = self.event_features is not None
        self.classified = run.classified

    def get_run_summaries(self):
        return [
            {
                "run_id": run.run_id,
                "biomarker_type": run.biomarker_type,
                "display_name": run.display_name,
                "detector_name": run.detector_name,
                "created_at": run.created_at,
                "accepted": run.run_id == self.analysis_session.accepted_run_id,
                "visible": self.analysis_session.is_run_visible(run.run_id),
                **run.summary,
            }
            for run in self.analysis_session.runs.values()
        ]

    def set_run_visible(self, run_id, visible):
        self.analysis_session.set_run_visible(run_id, visible)

    def get_visible_runs(self):
        return self.analysis_session.get_visible_runs()

    def accept_active_run(self):
        run = self.analysis_session.get_active_run()
        if run is None:
            return None
        self.analysis_session.accept_run(run.run_id)
        return run

    def get_decision_summary(self):
        active_run = self.analysis_session.get_active_run()
        accepted_run = self.analysis_session.get_accepted_run()
        ranking = self.get_channel_ranking(accepted_run.run_id if accepted_run else (active_run.run_id if active_run else None))
        top_channel = ranking[0] if ranking else None
        return {
            "num_runs": len(self.analysis_session.runs),
            "active_run_id": active_run.run_id if active_run else None,
            "active_detector": active_run.detector_name if active_run else None,
            "accepted_run_id": accepted_run.run_id if accepted_run else None,
            "accepted_detector": accepted_run.detector_name if accepted_run else None,
            "top_channel": top_channel,
        }

    def get_channel_ranking(self, run_id=None):
        return self.analysis_session.get_channel_ranking(run_id)

    def compare_runs(self, run_ids=None):
        return self.analysis_session.compare_runs(run_ids)

    def export_clinical_summary(self, path, run_id=None):
        run = self.analysis_session.get_run(run_id) if run_id else (self.analysis_session.get_accepted_run() or self.analysis_session.get_active_run())
        if run is None:
            return None
        ranking = pd.DataFrame(self.get_channel_ranking(run.run_id))
        comparison = pd.DataFrame(self.compare_runs().get("pairwise_overlap", []))
        runs = pd.DataFrame(self.get_run_summaries())
        accepted = self.analysis_session.get_accepted_run()
        accepted_meta = pd.DataFrame([{
            "active_run_id": self.analysis_session.active_run_id,
            "accepted_run_id": self.analysis_session.accepted_run_id,
            "accepted_detector": accepted.detector_name if accepted else "",
            "accepted_display_name": accepted.display_name if accepted else "",
            "exported_run_id": run.run_id,
            "exported_detector": run.detector_name,
        }])

        with pd.ExcelWriter(path) as writer:
            runs.to_excel(writer, sheet_name="Runs", index=False)
            ranking.to_excel(writer, sheet_name="Channel Ranking", index=False)
            comparison.to_excel(writer, sheet_name="Run Comparison", index=False)
            accepted_meta.to_excel(writer, sheet_name="Decision", index=False)
            if run.event_features is not None:
                run.event_features.to_df().to_excel(writer, sheet_name="Active Run Events", index=False)

    def to_checkpoint(self):
        checkpoint = build_base_checkpoint(self, self.biomarker_type)
        checkpoint["analysis_session"] = self.analysis_session.to_dict()
        checkpoint.update({
            "HFOs": self.HFOs,
            "param_detector": self.param_detector.to_dict() if self.param_detector else None,
            "event_features": self.event_features.to_dict() if self.event_features else None,
            "param_classifier": self.param_classifier.to_dict() if self.param_classifier else None,
            "artifact_predictions": np.array(self.event_features.artifact_predictions) if self.event_features else np.array([]),
            "spike_predictions": np.array(self.event_features.spike_predictions) if self.event_features else np.array([]),
            "ehfo_predictions": np.array(self.event_features.ehfo_predictions) if self.event_features else np.array([]),
            "artifact_annotations": np.array(self.event_features.artifact_annotations) if self.event_features else np.array([]),
            "pathological_annotations": np.array(self.event_features.pathological_annotations) if self.event_features else np.array([]),
            "physiological_annotations": np.array(self.event_features.physiological_annotations) if self.event_features else np.array([]),
            "annotated": np.array(self.event_features.annotated) if self.event_features else np.array([]),
        })
        return checkpoint

    def load_checkpoint(self, checkpoint):
        self.n_jobs = checkpoint_get(checkpoint, "n_jobs", self.n_jobs)
        self.eeg_data = checkpoint_array(checkpoint, "eeg_data")
        self.eeg_data_un60 = checkpoint_array(checkpoint, "eeg_data_un60", self.eeg_data)
        self.eeg_data_60 = checkpoint_array(checkpoint, "eeg_data_60", None)
        self.edf_param = checkpoint_get(checkpoint, "edf_param")
        self.sample_freq = checkpoint_get(checkpoint, "sample_freq", 0)
        self.channel_names = checkpoint_array(checkpoint, "channel_names")
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
            self.HFOs = checkpoint_array(checkpoint, "HFOs")
            detector_dict = checkpoint_get(checkpoint, "param_detector")
            if detector_dict:
                self.param_detector = ParamDetector.from_dict(detector_dict)
            event_dict = checkpoint_get(checkpoint, "event_features")
            if event_dict:
                self.event_features = HFO_Feature.from_dict(event_dict)
                self.event_features.artifact_predictions = checkpoint_array(checkpoint, "artifact_predictions", np.array([]))
                self.event_features.spike_predictions = checkpoint_array(checkpoint, "spike_predictions", np.array([]))
                self.event_features.ehfo_predictions = checkpoint_array(checkpoint, "ehfo_predictions", np.array([]))
                self.event_features.artifact_annotations = checkpoint_array(checkpoint, "artifact_annotations", np.array([]))
                self.event_features.pathological_annotations = checkpoint_array(checkpoint, "pathological_annotations", np.array([]))
                self.event_features.physiological_annotations = checkpoint_array(checkpoint, "physiological_annotations", np.array([]))
                self.event_features.annotated = checkpoint_array(checkpoint, "annotated", np.array([]))
                self.capture_current_run()
        
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
                biomarker_start, biomarker_end = start, min(end, self.sample_freq)
            elif end > len(data) - self.sample_freq // 2:
                plot_start, plot_end = len(data) - self.sample_freq, len(data)
                biomarker_start, biomarker_end = max(plot_start, start) - plot_start, min(plot_end, end) - plot_start
            else:
                plot_start, plot_end = (start + end)//2-self.sample_freq // 2, (start+end)//2+self.sample_freq // 2
                biomarker_start, biomarker_end = max(plot_start, start) - plot_start, min(plot_end, end) - plot_start
            plot_start, plot_end, biomarker_start, biomarker_end = int(plot_start), int(plot_end), int(biomarker_start), int(biomarker_end)
            channel_data = data[plot_start:plot_end]
            channel_data_f = data_filtered[plot_start:plot_end]
            #print(hfo_start, hfo_end, start, end, plot_start, plot_end, channel_data.shape, channel_data_f.shape)
            return channel_data, channel_data_f, biomarker_start, biomarker_end
        
        def extract_waveform(data, data_filtered, starts, ends, channel_names, unique_channel_names):
            biomarker_waveform_l, biomarker_waveform_f_l, biomarker_start_l , biomarker_end_l = np.zeros((len(starts), 2000)), np.zeros((len(starts), 2000)), [], []
            for i in tqdm(range(len(starts))):
                channel_name = channel_names[i]
                start = starts[i]
                end = ends[i]
                channel_index = np.where(unique_channel_names == channel_name)[0]
                biomarker_waveform, biomarker_waveform_f, biomarker_start, biomarker_end = extract_data(data[channel_index], data_filtered[channel_index], start, end)
                biomarker_waveform_l[i] = biomarker_waveform
                biomarker_waveform_f_l[i] = biomarker_waveform_f
                biomarker_start_l.append(biomarker_start)
                biomarker_end_l.append(biomarker_end)
            return biomarker_waveform_l, biomarker_waveform_f_l, np.array(biomarker_start_l), np.array(biomarker_end_l)

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
        start_s, end_s, feature_s, channel_names_s = starts[index_s], ends[index_s], feature[index_s], channel_names[index_s]
        index_a = np.where(spike_predictions == -1)[0]
        start_a, end_a, feature_a, channel_names_a = starts[index_a], ends[index_a], feature[index_a], channel_names[index_a]
        index_r = np.where(spike_predictions == 0)[0]
        start_r, end_r, feature_r, channel_names_r = starts[index_r], ends[index_r], feature[index_r], channel_names[index_r]
        #print("plotting HFO with spike")
        waveform_s, waveform_f_s, biomarker_start_s, biomarker_end_s = extract_waveform(self.eeg_data, self.filter_data, start_s, end_s, channel_names_s, self.channel_names)
        param_list = [{"folder": spike_folder, "start": start_s[i], "end": end_s[i], "feature": feature_s[i], "channel_name": channel_names_s[i], "data":waveform_s[i], "data_filtered":waveform_f_s[i], "hfo_start":biomarker_start_s[i], "hfo_end":biomarker_end_s[i]} for i in range(len(start_s))]
        ret = parallel_process(param_list, plot_feature, self.n_jobs, use_kwargs=True, front_num=3)
        waveform_a, waveform_f_a, biomarker_start_a, biomarker_end_a = extract_waveform(self.eeg_data, self.filter_data, start_a, end_a, channel_names_a, self.channel_names)
        param_list = [{"folder": artifact_folder, "start": start_a[i], "end": end_a[i], "feature": feature_a[i], "channel_name": channel_names_a[i], "data":waveform_a[i], "data_filtered":waveform_f_a[i], "hfo_start":biomarker_start_a[i], "hfo_end":biomarker_end_a[i]} for i in range(len(start_a))]
        ret = parallel_process(param_list, plot_feature, self.n_jobs, use_kwargs=True, front_num=3)
        waveform_r, waveform_f_r, biomarker_start_r, biomarker_end_r = extract_waveform(self.eeg_data, self.filter_data, start_r, end_r, channel_names_r, self.channel_names)
        param_list = [{"folder": non_spike_folder, "start": start_r[i], "end": end_r[i], "feature": feature_r[i], "channel_name": channel_names_r[i], "data":waveform_r[i], "data_filtered":waveform_f_r[i], "hfo_start":biomarker_start_r[i], "hfo_end":biomarker_end_r[i]} for i in range(len(start_r))]
        ret = parallel_process(param_list, plot_feature, self.n_jobs, use_kwargs=True, front_num=3)
