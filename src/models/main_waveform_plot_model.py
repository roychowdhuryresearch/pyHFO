import numpy as np
from src.hfo_app import HFO_App
from src.utils.utils_gui import *
from src.spindle_app import SpindleApp
import sys


class MainWaveformPlotModel:
    RUN_OVERLAY_COLORS = ["#4f7499", "#8a6b4f", "#6b8a5f", "#7f6592", "#5b8b84"]
    MIN_RENDER_POINTS = 1500

    def __init__(self, backend: HFO_App):
        self.backend = backend
        # self.color_dict={"artifact":(245,130,48), #orange
        #                  "spike":(240,30,250), #pink
        #                  "non_spike":(60,180,75), #green
        #                  "HFO":(60,180,75), #green
        #                  "waveform": (0,0,255),
        #                  }
        self.color_dict = COLOR_MAP
        
        self.start_in_time = 0
        self.end_in_time = 20
        self.time_window = 20 #20 second time window
        self.first_channel_to_plot = 0
        self.n_channels_to_plot = 10
        self.filtered = False
        self.normalize_vertical = False
        self._display_cache = {}
        self._overlay_cache = {}
        self._current_display_time = np.array([])
        self._current_full_window_data = None
        self._current_full_window_start_idx = 0
        self._current_full_window_key = None
        self.render_width_px = 1200
    
    def update_backend(self, new_backend):
        self.backend = new_backend
        self._display_cache = {}
        self._overlay_cache = {}
        self._current_display_time = np.array([])
        self._current_full_window_data = None
        self._current_full_window_start_idx = 0
        self._current_full_window_key = None

    def init_eeg_data(self):
        eeg_data, self.channel_names = self.backend.get_eeg_data()
        self.edf_info = self.backend.get_edf_info()
        self.sample_freq = self.edf_info['sfreq']
        self.n_samples = int(eeg_data.shape[1])
        self.total_time = max(0.0, (self.n_samples - 1) / self.sample_freq) if self.n_samples else 0.0

        self.filtered = False
        self.plot_biomarkers = False
        self.channel_names = list(self.channel_names)
        self.n_channels = len(self.channel_names)
        self.n_channels_to_plot = min(self.n_channels,self.n_channels_to_plot)
        self.channels_to_plot = self.channel_names.copy()
        self.channel_indices_to_plot = np.arange(self.n_channels)
        self._display_cache = {}
        self._overlay_cache = {}
        self._current_display_time = np.array([])
        self._current_full_window_data = None
        self._current_full_window_start_idx = 0
        self._current_full_window_key = None

    def set_time_window(self, time_window:int):
        self.time_window = time_window
        self._display_cache = {}
        self._overlay_cache = {}
        self._current_full_window_data = None
        self._current_full_window_key = None

    def set_plot_biomarkers(self, plot_biomarkers:bool):
        self.plot_biomarkers = plot_biomarkers

    def set_current_time_window(self, start_in_time):
        self.start_in_time = min(max(float(start_in_time), 0.0), self.total_time)
        self.end_in_time = min(self.start_in_time + self.time_window, self.total_time)
        if self.end_in_time <= self.start_in_time and self.n_samples > 1:
            self.end_in_time = min(self.start_in_time + (1.0 / self.sample_freq), self.total_time)

    def get_current_start_end(self):
        return self.start_in_time, self.end_in_time
    
    def get_current_time_window(self):
        return self._current_display_time

    def set_first_channel_to_plot(self, first_channel_to_plot):
        self.first_channel_to_plot = first_channel_to_plot

    def set_n_channels_to_plot(self, n_channels_to_plot:int):
        self.n_channels_to_plot = n_channels_to_plot
        self._display_cache = {}
        self._overlay_cache = {}
        self._current_full_window_data = None
        self._current_full_window_key = None

    def set_channels_to_plot(self, channels_to_plot:list):
        self.channels_to_plot = channels_to_plot
        self.channel_indices_to_plot = [self.channel_names.index(channel) for channel in channels_to_plot]
        self._display_cache = {}
        self._overlay_cache = {}
        self._current_full_window_data = None
        self._current_full_window_key = None

    def set_channel_indices_to_plot(self,channel_indices_to_plot:list):
        self.channel_indices_to_plot = channel_indices_to_plot
        self.channels_to_plot = [self.channel_names[index] for index in channel_indices_to_plot]
        self._display_cache = {}
        self._overlay_cache = {}
        self._current_full_window_data = None
        self._current_full_window_key = None

    def update_channel_names(self, new_channel_names):
        self.channel_names = list(new_channel_names)
        self._display_cache = {}
        self._overlay_cache = {}
        self._current_full_window_data = None
        self._current_full_window_key = None

    def set_waveform_filter(self, filtered):
        self.filtered = filtered
        self._display_cache = {}
        self._current_full_window_data = None
        self._current_full_window_key = None
    
    def set_normalize_vertical(self, normalize_vertical:bool):
        self.normalize_vertical = normalize_vertical
        self._display_cache = {}
        self._current_full_window_data = None
        self._current_full_window_key = None
        
    def get_waveform_color(self):
        return self.color_dict["waveform"]

    def get_total_time(self):
        return self.total_time

    def set_render_width_pixels(self, width_px):
        width_px = int(width_px or 0)
        if width_px <= 0:
            return
        self.render_width_px = width_px

    def _current_sample_bounds(self):
        start_idx = max(0, int(np.floor(self.start_in_time * self.sample_freq)))
        end_idx = min(self.n_samples, max(start_idx + 1, int(np.ceil(self.end_in_time * self.sample_freq))))
        return start_idx, end_idx

    def _build_time_axis(self, sample_indices):
        return np.asarray(sample_indices, dtype=np.float64) / float(self.sample_freq)

    def _target_render_points(self):
        return max(self.MIN_RENDER_POINTS, int(self.render_width_px * 2))

    def get_event_display_segment(self, channel_index, event_start_idx, event_end_idx):
        if self._current_full_window_data is None:
            return np.array([]), np.array([])
        start_rel = max(0, int(event_start_idx) - self._current_full_window_start_idx)
        end_rel = min(self._current_full_window_data.shape[1], int(event_end_idx) - self._current_full_window_start_idx)
        if end_rel <= start_rel:
            return np.array([]), np.array([])
        time_axis = self._build_time_axis(np.arange(int(event_start_idx), int(event_end_idx), dtype=np.int64))
        signal = self._current_full_window_data[channel_index, start_rel:end_rel]
        return time_axis, signal

    def _build_downsample_indices(self, eeg_data_to_display, start_idx):
        sample_count = eeg_data_to_display.shape[1]
        max_points = self._target_render_points()
        if sample_count <= max_points:
            return np.arange(start_idx, start_idx + sample_count, dtype=np.int64)

        bucket_count = max(1, max_points // 2)
        bucket_size = int(np.ceil(sample_count / bucket_count))
        amplitude = np.max(np.abs(eeg_data_to_display), axis=0) if eeg_data_to_display.shape[0] > 1 else np.abs(eeg_data_to_display[0])
        selected = []
        for bucket_start in range(0, sample_count, bucket_size):
            bucket_end = min(sample_count, bucket_start + bucket_size)
            selected.append(start_idx + bucket_start)
            segment = amplitude[bucket_start:bucket_end]
            if segment.size:
                peak_idx = start_idx + bucket_start + int(np.argmax(segment))
                if peak_idx != selected[-1]:
                    selected.append(peak_idx)
        last_idx = start_idx + sample_count - 1
        if selected[-1] != last_idx:
            selected.append(last_idx)
        return np.array(sorted(set(selected)), dtype=np.int64)

    def _normalize_window(self, eeg_data_to_display):
        if self.normalize_vertical:
            eeg_data_to_display = (eeg_data_to_display - eeg_data_to_display.min(axis=1, keepdims=True))
            max_value = np.max(eeg_data_to_display)
            if max_value > 0:
                eeg_data_to_display = eeg_data_to_display / max_value
            self.stds = 1
            return eeg_data_to_display

        means = np.mean(eeg_data_to_display)
        if self.filtered:
            self.stds = np.std(eeg_data_to_display) * 2
        else:
            self.stds = np.std(eeg_data_to_display)
        if self.stds == 0:
            self.stds = 1
        eeg_data_to_display = (eeg_data_to_display - means) / self.stds
        eeg_data_to_display[np.isnan(eeg_data_to_display)] = 0
        return eeg_data_to_display
    
    def get_all_current_eeg_data_to_display(self):
        start_idx, end_idx = self._current_sample_bounds()
        cache_key = (
            start_idx,
            end_idx,
            tuple(self.channel_indices_to_plot),
            self.filtered,
            self.normalize_vertical,
            self._target_render_points(),
        )
        cached = self._display_cache.get(cache_key)
        if cached is not None:
            self._current_display_time = cached[0]
            if self._current_full_window_key != cache_key:
                full_window_data, _ = self.backend.get_eeg_data(start_idx, end_idx, self.filtered)
                self._current_full_window_data = self._normalize_window(full_window_data[self.channel_indices_to_plot, :])
                self._current_full_window_start_idx = start_idx
                self._current_full_window_key = cache_key
            return cached[1:]

        eeg_data_to_display, _ = self.backend.get_eeg_data(start_idx, end_idx, self.filtered)
        eeg_data_to_display = eeg_data_to_display[self.channel_indices_to_plot,:]
        eeg_data_to_display = self._normalize_window(eeg_data_to_display)

        self._current_full_window_data = eeg_data_to_display.copy()
        self._current_full_window_start_idx = start_idx
        self._current_full_window_key = cache_key

        display_indices = self._build_downsample_indices(eeg_data_to_display, start_idx)
        eeg_data_to_display = eeg_data_to_display[:, display_indices - start_idx]
        time_to_display = self._build_time_axis(display_indices)

        #shift the ith channel by 1.1*i
        # eeg_data_to_display = eeg_data_to_display-1.1*np.arange(eeg_data_to_display.shape[0])[:,None]
        if self.filtered:
            # Add scale indicators
            # Set the length of the scale lines
            y_100_length = 50  # 100 microvolts
            offset_value = 6
            y_scale_length = y_100_length / self.stds
        else:
            y_100_length = 100  # 100 microvolts
            offset_value = 6
            y_scale_length = y_100_length / self.stds

        payload = (time_to_display, eeg_data_to_display, y_100_length, y_scale_length, offset_value)
        if len(self._display_cache) > 12:
            self._display_cache.pop(next(iter(self._display_cache)))
        self._display_cache[cache_key] = payload
        self._current_display_time = time_to_display
        return payload[1:]

    def _resolve_event_color(self, artifacts, spk_hfos, e_hfos, index, fallback_color):
        try:
            if artifacts[index] == -1:
                return self.color_dict["non_spike"]
            if int(artifacts[index]) < 1:
                return self.color_dict["artifact"]
            if spk_hfos[index]:
                return self.color_dict["spike"]
            if e_hfos[index]:
                return self.color_dict["ehfo"]
            return self.color_dict["non_spike"]
        except Exception:
            return fallback_color

    def _build_overlay_group(self, event_features, channel_in_name, group_color, is_active, order_index):
        if event_features is None:
            return None
        start_idx, end_idx = self._current_sample_bounds()
        cache_key = (
            id(event_features),
            channel_in_name,
            start_idx,
            end_idx,
            group_color,
            is_active,
            order_index,
        )
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        starts, ends, artifacts, spk_hfos, e_hfos = event_features.get_biomarkers_for_channel(
            channel_in_name,
            start_idx,
            end_idx,
        )
        colors = []

        for j in range(len(starts)):
            color = self._resolve_event_color(artifacts, spk_hfos, e_hfos, j, group_color) if is_active else group_color
            colors.append(color)

        starts_in_time = self._build_time_axis(starts)
        ends_in_time = self._build_time_axis(np.minimum(ends, max(self.n_samples - 1, 0)))
        payload = {
            "starts": starts,
            "ends": ends,
            "starts_in_time": starts_in_time,
            "ends_in_time": ends_in_time,
            "colors": colors,
            "line_width": 2 if is_active else 1.3,
            "marker_width": 10 if is_active else 6,
            "marker_offset": 0.18 * order_index,
        }
        if len(self._overlay_cache) > 128:
            self._overlay_cache.pop(next(iter(self._overlay_cache)))
        self._overlay_cache[cache_key] = payload
        return payload

    def get_all_biomarkers_for_all_current_channels_and_color(self, channel_in_name):
        session = getattr(self.backend, "analysis_session", None)
        if session is None or not hasattr(session, "get_visible_runs"):
            group = self._build_overlay_group(
                getattr(self.backend, "event_features", None),
                channel_in_name,
                self.color_dict["non_spike"],
                True,
                0,
            )
            return [group] if group is not None else []

        visible_runs = session.get_visible_runs()
        active_run_id = getattr(session, "active_run_id", None)
        overlay_groups = []
        for order_index, run in enumerate(visible_runs):
            is_active = run.run_id == active_run_id
            group_color = self.RUN_OVERLAY_COLORS[order_index % len(self.RUN_OVERLAY_COLORS)]
            group = self._build_overlay_group(run.event_features, channel_in_name, group_color, is_active, order_index)
            if group is not None:
                overlay_groups.append(group)
        return overlay_groups
