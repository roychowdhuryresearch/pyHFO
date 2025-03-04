class ParamClassifier():
    def __init__(self, artifact_path=None, spike_path=None, ehfo_path=None,
                 artifact_card=None, spike_card=None, ehfo_card=None,
                 use_spike=True, use_ehfo=True, device="cpu", batch_size=32, model_type="default_cpu"):
        self.artifact_path = artifact_path
        self.spike_path = spike_path
        self.ehfo_path = ehfo_path
        self.artifact_card = artifact_card
        self.spike_card = spike_card
        self.ehfo_card = ehfo_card
        self.use_spike = use_spike
        self.use_ehfo = use_ehfo
        self.device = device
        self.batch_size = batch_size
        self.model_type = model_type

    def to_dict(self):
        return {'artifact_path': self.artifact_path, 'spike_path': self.spike_path, 'ehfo_path': self.ehfo_path,
                'artifact_card': self.artifact_card, 'spike_card': self.spike_card, 'ehfo_card': self.ehfo_card,
                'use_spike': self.use_spike, 'use_ehfo': self.use_ehfo, 'device': self.device,
                'batch_size': self.batch_size, 'model_type': self.model_type}

    @staticmethod
    def from_dict(param_dict):
        return ParamClassifier(
            artifact_path=param_dict['artifact_path'],
            spike_path=param_dict['spike_path'],
            ehfo_path=param_dict['ehfo_path'],
            artifact_card=param_dict['artifact_card'],
            spike_card=param_dict['spike_card'],
            ehfo_card=param_dict['ehfo_card'],
            use_spike=param_dict['use_spike'],
            use_ehfo=param_dict['use_ehfo'],
            device=param_dict['device'],
            batch_size=param_dict['batch_size'],
            model_type=param_dict['model_type']
        )


class ParamModel():
    def __init__(self, in_channels=1, outputs=1, mode="None"):
        self.in_channels = in_channels
        self.outputs = outputs
        self.mode = mode

    def to_dict(self):
        return {'in_channels': self.in_channels, 'outputs': self.outputs}

    @staticmethod
    def from_dict(param_dict):
        return ParamModel(
            in_channels = param_dict['in_channels'],
            outputs = param_dict['outputs']
        )


class ParamPreprocessing():
    def __init__(self, image_size = 224, freq_range=[10, 500],
                 time_range=[0, 1000], fs=2000, crop_freq=[80, 224], crop_time=100):
        self.image_size = image_size
        self.freq_range = freq_range
        self.time_range = time_range
        self.fs = fs
        self.crop_freq = crop_freq
        self.crop_time = crop_time
        
    def to_dict(self):
        return {'image_size': self.image_size, 'freq_range': self.freq_range,
                'time_range': self.time_range, 'fs': self.fs,
                'crop_freq': self.crop_freq, 'crop_time': self.crop_time}

    @staticmethod
    def from_dict(param_dict):
        return ParamPreprocessing(
            image_size=param_dict['image_size'],
            freq_range=param_dict['freq_range_hz'],
            time_range=param_dict['time_range_ms'],
            fs=param_dict['fs'],
            crop_freq=param_dict['selected_freq_range_hz'],
            crop_time=param_dict['selected_window_size_ms'],
        )
