from multiprocessing import Pool
import matplotlib.pyplot as plt
from .utils import *
from scipy.signal import find_peaks, butter, sosfiltfilt, hilbert
import time



class SpikeDetector(object):
    def __init__(
        self,
        resample_rate=2000,  # hz
        window_size=5,  # seconds
        save_folder="results",
        ps_FreqSeg=128,
        ps_MinFreqHz=3,
        ps_MaxFreqHz=50,
        n_jobs=16,
        n_pre_spike=23,
        n_post_spike=51,
        threshold_factor=5,
        threshold_window=30,  # seconds
        filter_type="ellip",
        detect_mode="all",  # pos, neg, all
    ):
        self.param = {
            "resample_rate": float(resample_rate),  # hz
            "window_size": float(window_size),  # seconds
            "save_folder": save_folder,
            "ps_FreqSeg": float(ps_FreqSeg),
            "ps_MinFreqHz": float(ps_MinFreqHz),
            "ps_MaxFreqHz": float(ps_MaxFreqHz),
            "n_jobs": int(n_jobs),
            "n_pre_spike": int(n_pre_spike),
            "n_post_spike": int(n_post_spike),
            "threshold_factor": float(threshold_factor),
            "threshold_window": float(threshold_window),  # seconds
            "filter_type": filter_type,
            "detect_mode": detect_mode,  # pos, neg, all
        }
        self.sample_ref = int(1.5 * self.param["resample_rate"] / 1000) // 2
        self.save_folder = None

    def quick_view(self, data):
        t = np.arange(len(data))
        plt.plot(t, data)
        plt.savefig("test_view.jpg")

    def calculate_boundary(self, mid, window_size, signal_length):
        sample_before = int(self.param["n_pre_spike"])
        sample_after = int(self.param["n_post_spike"])
        if mid - window_size // 2 < 0:
            s = 0
            e = window_size
        elif mid + window_size // 2 >= signal_length:
            s = signal_length - window_size
            e = signal_length
        else:
            s = mid - window_size // 2
            e = mid + window_size // 2
        event_start = max(0, mid - sample_before - s)
        event_end = min(signal_length, mid + sample_after - s)
        return int(s), int(e), int(event_start), int(event_end)

    def get_threshold(self, data):
        threshold = np.zeros(len(data))
        for i in range(
            round(
                len(data) / self.param["threshold_window"] / self.param["resample_rate"]
            )
            + 1
        ):
            start = min(
                i * self.param["threshold_window"] * self.param["resample_rate"],
                len(data),
            )
            end = min(
                (i + 1) * self.param["threshold_window"] * self.param["resample_rate"],
                len(data),
            )

            start, end = int(start), int(end)

            signal = data[start:end]
            if not signal.any():
                continue

            analytic_signal = hilbert(signal)
            envelope = np.abs(analytic_signal)
            # lmin, lmax = hl_envelopes_idx(signal)

            # new = np.median(np.abs(np.concatenate((signal[lmax], signal[lmin])))) / 0.6745
            new = np.median(np.abs(envelope)) / 0.6745
            # old = np.median(np.abs(signal)) / 0.6745
            threshold[start:end] = self.param["threshold_factor"] * new
        return threshold

    def get_threshold2(self, data):
        threshold = np.zeros(len(data))
        for i in range(
            round(
                len(data) / self.param["threshold_window"] / self.param["resample_rate"]
            )
            + 1
        ):
            start = min(
                i * self.param["threshold_window"] * self.param["resample_rate"],
                len(data),
            )
            end = min(
                (i + 1) * self.param["threshold_window"] * self.param["resample_rate"],
                len(data),
            )

            signal = data[start:end]
            if not signal.any():
                continue
            lmin, lmax = hl_envelopes_idx(signal)

            new = (
                np.median(np.abs(np.concatenate((signal[lmax], signal[lmin])))) / 0.6745
            )
            # old = np.median(np.abs(signal)) / 0.6745
            threshold[start:end] = self.param["threshold_factor"] * new
        return threshold

    def spikes_detect(self, data):
        original_data = np.copy(data)
        filter_range = 50  # [58., 62.]
        sos = butter(20, filter_range, btype="lowpass", output="sos", fs=2000)
        original_data = sosfiltfilt(sos, original_data)

        # git rid of the noise
        filter_range = [58.0, 62.0]
        sos = butter(
            20,
            filter_range,
            btype="bandstop",
            output="sos",
            fs=self.param["resample_rate"],
        )
        filtered_data = sosfiltfilt(sos, data)
        # thres_lower_bound = self.get_threshold(abs(filtered_data))
        # 1. filter the signal
        if self.param["filter_type"] == "cheby2":
            ft = FilterCheby2(25, 80, 0.5, 93, 0.5, self.param["resample_rate"])
            filtered_data = ft.filter_data(filtered_data)
        elif self.param["filter_type"] == "ellip":
            ft = FilterEllip(
                order=4,
                rp=0.1,
                rs=40,
                wn=(25, 80),
                btype="bandpass",
                fs=self.param["resample_rate"],
            )
            filtered_data = ft.filter_data(filtered_data)

        # 2. rectification
        # rectified_data = abs(filtered_data)
        thres = self.get_threshold(abs(filtered_data))
        distance = 0.5 * self.param["resample_rate"]

        if self.param["detect_mode"] == "all":
            peaks = find_peaks(abs(filtered_data), height=thres, distance=distance)[0]
        if self.param["detect_mode"] == "pos":
            peaks = find_peaks(filtered_data, height=thres, distance=distance)[0]
        if self.param["detect_mode"] == "neg":
            peaks = find_peaks(
                np.negative(filtered_data), height=thres, distance=distance
            )[0]

        # reduced_peaks_mask = filtered_data[peaks] > thres_lower_bound[peaks]
        # reduced_peaks = peaks[reduced_peaks_mask]
        # return original_data, filtered_data, thres[reduced_peaks], reduced_peaks
        return original_data, filtered_data, thres[peaks], peaks

    def detect_spikes_one_channel(self, data, channel_name):
        results = self.spikes_detect(data)
        if not results[-1].any():
            return [], [], [], []
        signals_original = results[0]
        signals = results[1]
        thresholds = results[2]
        peaks = results[3]

        window_size = int(self.param["window_size"] * self.param["resample_rate"])

        waveform_ori = np.zeros((len(peaks), window_size))
        waveform = np.zeros((len(peaks), window_size))
        spike_starts = np.zeros(len(peaks))
        spike_ends = np.zeros(len(peaks))
        window_starts = np.zeros(len(peaks))
        window_ends = np.zeros(len(peaks))

        for i, p in enumerate(peaks):
            ws, we, ss, se = self.calculate_boundary(p, window_size, len(signals))
            waveform_ori[i] = signals_original[ws:we]
            waveform[i] = signals[ws:we]
            window_starts[i] = ws
            window_ends[i] = we
            spike_starts[i] = ss
            spike_ends[i] = se

        return (
            waveform_ori,
            waveform,
            spike_starts,
            spike_ends,
            np.array([channel_name] * len(waveform), dtype=object),
            window_starts,
            window_ends,
            thresholds,
        )

    def detect_multi_channels(self, data, channel_names, filtered=True):
    # def detect_spikes_multi_channels(self, data, channel_names):
        (
            waveform_ori,
            waveform,
            starts,
            ends,
            channels,
            window_starts,
            window_ends,
            thresholds,
        ) = ([], [], [], [], [], [], [], [])
        with Pool(self.param["n_jobs"]) as p:
            results = p.starmap(
                self.detect_spikes_one_channel,
                [(data[i, :], channel_names[i]) for i in range(data.shape[0])],
            )
        # results = []
        # for i in range(data.shape[0]):
        #     results.append(self.detect_spikes_one_channel(data[i, :], channel_names[i]))

        for result in results:
            if len(result[0]) == 0:
                continue
            waveform_ori.append(result[0])
            waveform.append(result[1])
            starts.append(result[2])
            ends.append(result[3])
            channels.append(result[4])
            window_starts.append(result[5])
            window_ends.append(result[6])
            thresholds.append(result[7])

        starts = np.concatenate(starts)
        ends = np.concatenate(ends)
        spikes = np.concatenate((starts.reshape(-1, 1), ends.reshape(-1, 1)), axis=1)
        return (
            # np.concatenate(waveform_ori, axis=0),
            # np.concatenate(waveform, axis=0),
            # np.concatenate(starts),
            # np.concatenate(ends),
            np.concatenate(channels),
            spikes,
            # np.concatenate(window_starts),
            # np.concatenate(window_ends),
            # np.concatenate(thresholds),
        )

    def plot_time_frequency(
        self, data_ori, data, channel_name, start, end, plt_s, plt_e, threshold
    ):
        # time_freqency_plot = compute_spectrum(data, self.param["resample_rate"], self.param["ps_FreqSeg"],
        #                                       self.param["ps_MinFreqHz"], self.param["ps_MaxFreqHz"])
        fig, axs = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        plt_s, plt_e, start, end = int(plt_s), int(plt_e), int(start), int(end)
        index = np.arange(len(data))
        axs[0].plot(index, data_ori, c="blue")
        axs[1].plot(index, data, c="blue")
        axs[1].axhline(y=threshold, color="g", linestyle="-")
        axs[1].axhline(y=-threshold, color="g", linestyle="-")
        spike_index = index[start:end]
        axs[1].plot(spike_index, data[spike_index], c="red")
        axs[1].plot(len(data) // 2, data[len(data) // 2], "x")
        # axs[2].set_xticks(np.linspace(0, len(data), 5))
        # axs[2].set_xticklabels(np.round(np.linspace(plt_s, plt_e, 5) / self.param["resample_rate"], 3))
        # axs[2].set_xlabel("second")
        # axs[2].imshow(time_freqency_plot, aspect='auto')
        # axs[2].set_yticks(np.linspace(0, self.param["ps_FreqSeg"], 5))
        # axs[2].set_yticklabels(np.linspace(self.param["ps_MinFreqHz"], self.param["ps_MaxFreqHz"], 5)[::-1].astype(int))
        axs[1].set_xticks(np.linspace(0, len(data), 5))
        axs[1].set_xticklabels(
            np.round(np.linspace(plt_s, plt_e, 5) / self.param["resample_rate"], 3)
        )
        axs[1].set_xlabel("second")

        filename = self.param["edf_fn"]
        title = f"{filename}: {channel_name}_{start}_{end}"
        plt.suptitle(title)
        img_name = f"{channel_name}_{plt_s}_{plt_e}"
        save_folder = os.path.join(self.save_folder, channel_name)
        os.makedirs(save_folder, exist_ok=True)
        fn = os.path.join(save_folder, img_name + ".jpg")
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()

    def plot_time_frequency_multi_process(
        self,
        waveforms_ori,
        waveforms,
        channel_names,
        starts,
        ends,
        plt_s,
        plt_e,
        thresholds,
    ):
        param_list = [
            (
                waveforms_ori[i],
                waveforms[i],
                channel_names[i],
                starts[i],
                ends[i],
                plt_s[i],
                plt_e[i],
                thresholds[i],
            )
            for i in range(len(waveforms))
        ]
        with Pool(self.param["n_jobs"]) as p:
            p.starmap(self.plot_time_frequency, param_list)
        # for i in range(len(waveforms)):
        #     self.plot_time_frequency(waveforms_ori[i], waveforms[i], channel_names[i],
        #                              starts[i], ends[i], plt_s[i], plt_e[i], thresholds[i])

    def pipeline(self):
        data, channel_names = read_raw(
            "AG_A_ave.edf", resample=2000
        )
        # # crop the data for test purpose
        # data = data[:, 0:self.param['resample_rate'] * 1800]
        begin = time.time()
        (
            spikes,
            channel_names
        ) = self.detect_multi_channels(data, channel_names)
        end = time.time()
        print(end - begin)
        
        return (
            # waveforms_ori,
            # waveforms,
            spikes,
            channel_names
            # plt_starts,
            # plt_ends,
            # thresholds,
        )

    def test_purpose(self):
        data, channel_names = read_raw(
            "AG_A_ave.edf", resample=2000
        )
        channel_names = channel_names[27]
        data = data[27, 668000:670000]
        results = self.detect_spikes_one_channel(data, channel_names)


if __name__ == "__main__":
    detector = SpikeDetector()
    detector.pipeline()
    # detector.test_purpose()
