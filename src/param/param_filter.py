class ParamFilter:
    def __init__(self, fp=80, fs=500, rp=0.5, rs=93, space=0.5, sample_freq=2000):
        self.fp = fp
        self.fs = fs
        self.rp = rp
        self.rs = rs
        self.space = space
        self.sample_freq = sample_freq
    def to_dict(self):
        return {'fp':self.fp, 'fs':self.fs, 'rp':self.rp, 'rs':self.rs, 'space':self.space, 'sample_freq':self.sample_freq}

    @staticmethod
    def from_dict(param_filter):
        if not 'sample_freq' in param_filter:
            param_filter['sample_freq'] = None
        return ParamFilter(
            fp = param_filter['fp'],
            fs = param_filter['fs'],
            rp = param_filter['rp'],
            rs = param_filter['rs'],
            sample_freq = param_filter['sample_freq']
        )