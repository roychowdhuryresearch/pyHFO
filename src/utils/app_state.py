import numpy as np

APP_STATE_VERSION = 3


def _item_or_value(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def checkpoint_get(checkpoint, key, default=None):
    if key not in checkpoint:
        return default
    return _item_or_value(checkpoint[key])


def checkpoint_array(checkpoint, key, default=None):
    if key not in checkpoint:
        return default
    return checkpoint[key]


def build_base_checkpoint(app, biomarker_type):
    return {
        "app_state_version": APP_STATE_VERSION,
        "biomarker_type": biomarker_type,
        "n_jobs": app.n_jobs,
        "eeg_data": app.eeg_data,
        "eeg_data_un60": app.eeg_data_un60,
        "eeg_data_60": app.eeg_data_60,
        "edf_param": app.edf_param,
        "sample_freq": app.sample_freq,
        "channel_names": app.channel_names,
        "recording_channel_names": getattr(app, "recording_channel_names", None),
        "param_filter": app.param_filter.to_dict() if app.param_filter else None,
        "classified": app.classified,
        "filtered": app.filtered,
        "detected": app.detected,
        "filter_data": app.filter_data,
        "filter_data_un60": app.filter_data_un60,
        "filter_data_60": app.filter_data_60,
    }


def checkpoint_version(checkpoint):
    return checkpoint_get(checkpoint, "app_state_version", 1)


def is_legacy_checkpoint(checkpoint):
    return checkpoint_version(checkpoint) < APP_STATE_VERSION


def empty_array(length=0):
    return np.zeros(length) if length else np.array([])
