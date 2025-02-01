import numpy as np
from src.utils.utils_inference import inference, load_model, load_ckpt
from src.param.param_classifier import ParamClassifier
import torch
from src.model import PreProcessing
from transformers import TrainingArguments, ViTForImageClassification
from transformers import Trainer
from src.dl_models import *
import os


class Classifier():
    def __init__(self, param:ParamClassifier):
        self.device = param.device
        self.batch_size = param.batch_size
        self.model_type = param.model_type
        self.use_spike = param.use_spike
        self.use_ehfo = param.use_ehfo
        self.load_func = torch.load if "default" in self.model_type else torch.load  #torch.hub.load_state_dict_from_url 
        if param.artifact_path:
            self.update_model_a(param)
            self.update_model_toy(param)
        if param.spike_path:
            self.update_model_s(param)
        if param.ehfo_path:
            self.update_model_e(param)
        
    def update_model_s(self, param:ParamClassifier):
        self.model_type = param.model_type
        self.spike_path = param.spike_path
        self.param_spike_preprocessing, self.model_s = load_ckpt(self.load_func, param.spike_path)
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

    def update_model_e(self, param:ParamClassifier):
        self.model_type = param.model_type
        self.ehfo_path = param.ehfo_path
        self.param_ehfo_preprocessing, self.model_e = load_ckpt(self.load_func, param.ehfo_path)
        if self.model_type == "default_cpu":
            param.device = "cpu"
        elif self.model_type == "default_gpu":
            param.device = "cuda:0"
        else:
            raise ValueError("Model type not supported!")
        self.device = param.device if torch.cuda.is_available() else "cpu"
        self.model_e = self.model_e.to(self.device)
        self.preprocessing_ehfo = PreProcessing.from_param(self.param_ehfo_preprocessing)

    def update_model_toy(self, param:ParamClassifier):
        self.model_type = param.model_type
        self.artifact_path = param.artifact_path
        res_dir = os.path.dirname(param.artifact_path)
        model = NeuralCNNForImageClassification.from_pretrained(os.path.join(res_dir, 'model_toy'))

        self.param_artifact_preprocessing, _ = load_ckpt(self.load_func, param.artifact_path)
        if "default" in self.model_type:
            model.channel_selection = True
            model.input_channels = 1
        if self.model_type == "default_cpu":
            param.device = "cpu"
        elif self.model_type == "default_gpu":
            param.device = "cuda:0"
        else:
            raise ValueError("Model type not supported!")
        self.device = param.device if torch.cuda.is_available() else "cpu"
        self.model_toy = model.to(self.device)
        self.preprocessing_artifact = PreProcessing.from_param(self.param_artifact_preprocessing)

    def artifact_detection(self, biomarker_features, ignore_region, threshold=0.5):
        if not self.model_toy:
            raise ValueError("Please load artifact model first!")
        return self._classify_artifacts(self.model_toy, biomarker_features, ignore_region, threshold=threshold)

    def spike_detection(self, biomarker_features):
        if not self.model_s:
            raise ValueError("Please load spike model first!")
        return self._classify_spikes(self.model_s, biomarker_features)

    def ehfo_detection(self, biomarker_features):
        if not self.model_e:
            raise ValueError("Please load eHFO model first!")
        return self._classify_ehfos(self.model_s, biomarker_features)

    def _classify_artifacts(self, model, biomarker_feature, ignore_region, threshold=0.5):
        model = model.to(self.device)
        features = self.preprocessing_artifact.process_biomarker_feature(biomarker_feature)
        artifact_predictions = np.zeros(features.shape[0]) -1
        starts = biomarker_feature.starts
        ends = biomarker_feature.ends
        keep_index = np.where(np.logical_and(starts > ignore_region[0], ends < ignore_region[1]) == True)[0]   
        features = features[keep_index]
        if len(features) != 0:
            predictions = inference(model, features, self.device, self.batch_size, threshold=threshold)
            artifact_predictions[keep_index] = predictions
        biomarker_feature.update_artifact_pred(artifact_predictions)
        return biomarker_feature
    
    def _classify_spikes(self, model, biomarker_feature):
        if len(biomarker_feature.artifact_predictions) == 0:
            raise ValueError("Please run artifact classifier first!")
        model = model.to(self.device)
        features = self.preprocessing_spike.process_biomarker_feature(biomarker_feature)
        spike_predictions = np.zeros(features.shape[0]) -1
        keep_index = np.where(biomarker_feature.artifact_predictions > 0)[0]
        features = features[keep_index]
        if len(features) != 0:
            predictions = inference(model, features, self.device, self.batch_size)
            spike_predictions[keep_index] = predictions
        biomarker_feature.update_spike_pred(spike_predictions)
        return biomarker_feature

    def _classify_ehfos(self, model, biomarker_feature):
        if len(biomarker_feature.artifact_predictions) == 0:
            raise ValueError("Please run artifact classifier first!")
        model = model.to(self.device)
        features = self.preprocessing_ehfo.process_biomarker_feature(biomarker_feature)
        ehfo_predictions = np.zeros(features.shape[0]) -1
        keep_index = np.where(biomarker_feature.artifact_predictions > 0)[0]
        features = features[keep_index]
        if len(features) != 0:
            predictions = inference(model, features, self.device, self.batch_size)
            ehfo_predictions[keep_index] = predictions
        biomarker_feature.update_ehfo_pred(ehfo_predictions)
        return biomarker_feature


if __name__ == '__main__':
    model_type = 'default_cpu'
    ehfo_path = 'E:\\projects\\pyHFO\\ckpt\\model_e.tar'
    param_ehfo_preprocessing, model_e = load_ckpt(torch.load, ehfo_path)
    device = "cpu"
    model_e = model_e.to(device)
    preprocessing_ehfo = PreProcessing.from_param(param_ehfo_preprocessing)