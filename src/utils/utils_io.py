import os
import re
from pathlib import Path

import mne
import numpy as np


def load_mne_raw(file_path):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".edf":
        return mne.io.read_raw_edf(file_path, verbose=0)
    if suffix == ".vhdr":
        _assert_brainvision_triplet(file_path)
        return mne.io.read_raw_brainvision(file_path, verbose=0)
    if suffix == ".eeg":
        header_path = file_path.replace(".eeg", ".vhdr")
        _assert_brainvision_triplet(header_path)
        return mne.io.read_raw_brainvision(header_path, verbose=0)
    if suffix == ".vmrk":
        header_path = file_path.replace(".vmrk", ".vhdr")
        _assert_brainvision_triplet(header_path)
        return mne.io.read_raw_brainvision(header_path, verbose=0)
    if suffix == ".fif" or file_path.lower().endswith(".fif.gz"):
        return mne.io.read_raw_fif(file_path, verbose=0)
    raise ValueError(f"File type not supported: {suffix}")


def _assert_brainvision_triplet(header_path):
    eeg_path = header_path.replace(".vhdr", ".eeg")
    vmrk_path = header_path.replace(".vhdr", ".vmrk")
    if not os.path.exists(eeg_path):
        raise AssertionError("The .eeg file does not exist, cannot load the data")
    if not os.path.exists(vmrk_path):
        raise AssertionError("The .vmrk file does not exist, cannot load the data")

def get_edf_info(raw):
    res = {
            "channels": raw.info['ch_names'],
            "sfreq": raw.info['sfreq'],
            "nchan": raw.info['nchan'],
            "meas_date": raw.info['meas_date'],
            "highpass": raw.info['highpass'],
            "lowpass": raw.info['lowpass'],
        }
    return res

def read_eeg_data(raw):
    raw_channels = raw.info['ch_names']
    eeg_data = np.asarray(raw.get_data()) * 1E6
    if eeg_data.ndim == 1:
        eeg_data = np.expand_dims(eeg_data, axis=0)
    channel_names = np.array(raw_channels)
    #sort channel_names
    indices, channel_names = sort_channel(channel_names)
    eeg_data = eeg_data[indices]
    return eeg_data, channel_names

def dump_to_npz(data, fn):
    # import pickle
    # pickle.dump(data, open(fn, 'wb'))
    np.savez_compressed(fn, **data)
    # import torch
    # torch.save(data, fn)

def sort_filename(filename):
    """Extract the numeric part of the filename and use it as the sort key"""
    return [int(x) if x.isdigit() else x for x in re.findall(r'\d+|\D+', filename)]

def sort_channel(channel_names):
    def get_key(item):
        try:
            index, channel = item
            if channel.startswith('POL'):
                # For strings starting with "POL", split into prefix, mid-part, and number
                prefix, mid, number = re.search(r'(POL) (\w+)(\d+)', channel).groups()
                return (1, prefix, mid, int(number))
            else:
                # For strings ending with "Ref", "REF", or "ref", extract the letter, number, and case
                match = re.match(r'([A-Z][a-z]*)(\d+)(Ref|REF|ref)', channel)
                if match:
                    letter, number, ref = match.groups()
                    return (2, letter, int(number), ref.lower())
                else:
                    # If no match, return a high sort key to sort these items last
                    return (3, sort_filename(channel))
        except:
            return (3, sort_filename(channel))

    sorted_channels = sorted(enumerate(channel_names), key=get_key)
    
    # Unpack the indices and names into separate lists
    indices, names = zip(*sorted_channels)
    
    return np.array(indices), np.array(names)
