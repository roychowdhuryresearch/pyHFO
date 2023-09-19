import numpy as np
from src.utils.utils_inference import inference, load_model, load_ckpt
from src.param.param_classifier import ParamClassifier
import torch
from src.model import PreProcessing
class Classifier():
    def __init__(self, param:ParamClassifier):
        self.device = param.device
        self.batch_size = param.batch_size
        self.model_type = param.model_type
        self.use_spike = param.use_spike
        self.load_func = torch.load if "default" in self.model_type else torch.load  #torch.hub.load_state_dict_from_url 
        if param.artifact_path:
            self.update_model_a(param)
        if param.spike_path:
            self.update_model_s(param)
        
    def update_model_s(self, param:ParamClassifier):
        self.model_type = param.model_type
        self.spike_path = param.spike_path
        self.param_spike_preprocessing, self.model_s = load_ckpt(self.load_func , param.spike_path)
        #print("spike model param", self.param_spike.to_dict())
        if self.model_type == "default_cpu":
            param.device = "cpu"
        elif self.model_type == "default_gpu":
            param.device = "cuda:0"
        else:
            raise ValueError("Model type not supported!")
        self.device = param.device if torch.cuda.is_available() else "cpu"
        self.model_s = self.model_s.to(self.device)
        self.preprocessing_spike = PreProcessing.from_param(self.param_spike_preprocessing)

    def update_model_a(self, param:ParamClassifier):
        self.model_type = param.model_type
        self.artifact_path = param.artifact_path
        self.param_artifact_preprocessing, model = load_ckpt(self.load_func, param.artifact_path)
        if "default" in self.model_type:
            model.channel_selection = True
            model.in_channels = 1
        if self.model_type == "default_cpu":
            param.device = "cpu"
        elif self.model_type == "default_gpu":
            param.device = "cuda:0"
        else:
            raise ValueError("Model type not supported!")
        self.device = param.device if torch.cuda.is_available() else "cpu"
        self.model_a = model.to(self.device)
        self.preprocessing_artifact = PreProcessing.from_param(self.param_artifact_preprocessing)

    def artifact_detection(self, HFO_features, ignore_region, threshold=0.5):
        if not self.model_a:
            raise ValueError("Please load artifact model first!")
        return self._classify_artifacts(self.model_a, HFO_features, ignore_region, threshold=threshold)

    def spike_detection(self, HFO_features):
        if not self.model_s:
            raise ValueError("Please load spike model first!")
        return self._classify_spikes(self.model_s, HFO_features)

    def _classify_artifacts(self, model, HFO_feature, ignore_region, threshold=0.5):
        model = model.to(self.device)
        features = self.preprocessing_artifact.process_hfo_feature(HFO_feature)
        artifact_predictions = np.zeros(features.shape[0]) -1
        starts = HFO_feature.starts
        ends = HFO_feature.ends
        keep_index = np.where(np.logical_and(starts > ignore_region[0], ends < ignore_region[1]) == True)[0]   
        features = features[keep_index]
        if len(features) != 0:
            predictions = inference(model, features, self.device ,self.batch_size, threshold=threshold)
            artifact_predictions[keep_index] = predictions
        HFO_feature.update_artifact_pred(artifact_predictions)
        return HFO_feature
    
    def _classify_spikes(self, model, HFO_feature):
        if len(HFO_feature.artifact_predictions) == 0:
            raise ValueError("Please run artifact classifier first!")
        model = model.to(self.device)
        features = self.preprocessing_spike.process_hfo_feature(HFO_feature)
        spike_predictions = np.zeros(features.shape[0]) -1
        keep_index = np.where(HFO_feature.artifact_predictions > 0)[0]
        features = features[keep_index]
        if len(features) != 0:
            predictions = inference(model, features, self.device, self.batch_size)
            spike_predictions[keep_index] = predictions
        HFO_feature.update_spike_pred(spike_predictions)
        return HFO_feature

