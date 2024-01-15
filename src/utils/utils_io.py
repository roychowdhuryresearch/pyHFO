import numpy as np
import re

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
    data = []

    for raw_ch in raw_channels:
        ch_data = raw.get_data(raw_ch) * 1E6
        data.append(ch_data)

    eeg_data = np.squeeze(data)
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
                    return (3, channel)
        except:
            return (3, channel)

    sorted_channels = sorted(enumerate(channel_names), key=get_key)
    
    # Unpack the indices and names into separate lists
    indices, names = zip(*sorted_channels)
    
    return np.array(indices), np.array(names)