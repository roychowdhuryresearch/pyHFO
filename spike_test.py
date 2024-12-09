import mne
import numpy as np
from scipy.signal import hilbert, find_peaks
from scipy.signal import detrend

# Load the EDF file using MNE
file_path = 'MV_2.edf'  # Replace with your actual file path
raw_data = mne.io.read_raw_edf(file_path, preload=True)

# Get basic information about the file
info = raw_data.info
signal_labels = raw_data.ch_names
sampling_rate = raw_data.info['sfreq']

# Apply notch filter to remove 60 Hz noise
raw_notched = raw_data.copy().notch_filter(freqs=60, method='fir', fir_design='firwin')

# Bandpass filter the data between 25 and 80 Hz using MNE's built-in FIR filter
low_cutoff = 25  # Lower bound of the bandpass filter in Hz
high_cutoff = 80  # Upper bound of the bandpass filter in Hz
raw_filtered = raw_notched.copy().filter(l_freq=low_cutoff, h_freq=high_cutoff, method='fir', fir_design='firwin', phase='zero-double')


# Extract the filtered signal and unfiltered signal for the first channel
filtered_signal_data, filtered_times = raw_filtered[0, :]
unfiltered_signal_data, _ = raw_notched[0, :]

# Apply Hilbert transform to the filtered signal
analytic_signal = hilbert(filtered_signal_data[0])

# Calculate the amplitude envelope (magnitude of the analytic signal)
amplitude_envelope = np.abs(analytic_signal)

# Calculate the mean amplitude of the envelope
mean_amplitude = np.mean(amplitude_envelope)

# Identify candidate spikes where the amplitude envelope exceeds 3 times the mean amplitude
threshold = 3 * mean_amplitude
candidate_spikes = np.where(amplitude_envelope > threshold)[0]

# Time window around each candidate spike (±0.25 s)
window_size = int(0.25 * sampling_rate)

# List to store the valid spikes
valid_spikes = []

for spike_idx in candidate_spikes:
    # Get the time window around the spike for unfiltered data
    start_idx = max(0, spike_idx - window_size)
    end_idx = min(len(unfiltered_signal_data[0]), spike_idx + window_size)

    signal_window = unfiltered_signal_data[0][start_idx:end_idx]

    # Detrend the window
    detrended_signal = detrend(signal_window)

    # Identify peaks and troughs
    peaks, _ = find_peaks(detrended_signal)
    troughs, _ = find_peaks(-detrended_signal)

    # Calculate the Fano factor
    if len(peaks) > 1 and len(troughs) > 1:
        # Calculate inter-peak and inter-trough intervals
        inter_peak_intervals = np.diff(peaks) / sampling_rate  # Convert to seconds
        inter_trough_intervals = np.diff(troughs) / sampling_rate

        peak_fano_factor = np.var(inter_peak_intervals) / np.mean(inter_peak_intervals)
        trough_fano_factor = np.var(inter_trough_intervals) / np.mean(inter_trough_intervals)

        fano_factor = (peak_fano_factor + trough_fano_factor) / 2  # Average

        # Calculate the maximum amplitude in the window (rectified)
        max_amplitude = np.max(np.abs(signal_window))

        # Check the conditions for valid spikes
        if max_amplitude > 3 * mean_amplitude and fano_factor >= 2.5:
            valid_spikes.append(spike_idx)

# Merge spikes detected within 20 ms of each other
merged_spikes = []
previous_spike = -np.inf
for spike in valid_spikes:
    if spike - previous_spike > 0.02 * sampling_rate:
        merged_spikes.append(spike)
    previous_spike = spike

# Display final spike indices and times
spike_times = filtered_times[merged_spikes]
print(f"Detected spikes at times (s): {spike_times}")
