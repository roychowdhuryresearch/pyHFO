import numpy as np
from src.hfo_app import HFO_App
from src.utils.utils_gui import *
from src.spindle_app import SpindleApp
import sys


class MiniPlotModel:
    RUN_OVERLAY_COLORS = ["#4f7499", "#8a6b4f", "#6b8a5f", "#7f6592", "#5b8b84"]

    def __init__(self, backend: HFO_App):
        self.backend = backend
        self.overlay_run_provider = None
        # self.color_dict={"artifact":(245,130,48), #orange
        #                  "spike":(240,30,250), #pink
        #                  "non_spike":(60,180,75), #green
        #                  "HFO":(60,180,75), #green
        #                  }
        self.color_dict = COLOR_MAP
        self.first_channel_to_plot = 0
        self.n_channels_to_plot = 10

    def update_backend(self, new_backend):
        self.backend = new_backend

    def set_overlay_run_provider(self, provider):
        self.overlay_run_provider = provider

    def init_eeg_data(self):
        eeg_data, self.channel_names = self.backend.get_eeg_data()
        self.edf_info = self.backend.get_edf_info()
        self.sample_freq = self.edf_info['sfreq']
        self.n_samples = int(eeg_data.shape[1])
        self.total_time = max(0.0, (self.n_samples - 1) / self.sample_freq) if self.n_samples else 0.0

        self.channel_names = list(self.channel_names)
        self.n_channels = len(self.channel_names)
        self.n_channels_to_plot = min(self.n_channels,self.n_channels_to_plot)
        self.channels_to_plot = self.channel_names.copy()
        self.channel_indices_to_plot = np.arange(self.n_channels)

    def set_first_channel_to_plot(self, first_channel_to_plot):
        self.first_channel_to_plot = first_channel_to_plot

    def set_n_channels_to_plot(self, n_channels_to_plot:int):
        self.n_channels_to_plot = n_channels_to_plot

    def set_channels_to_plot(self, channels_to_plot:list):
        self.channels_to_plot = channels_to_plot
        self.channel_indices_to_plot = [self.channel_names.index(channel) for channel in channels_to_plot]

    def set_channel_indices_to_plot(self,channel_indices_to_plot:list):
        self.channel_indices_to_plot = channel_indices_to_plot
        self.channels_to_plot = [self.channel_names[index] for index in channel_indices_to_plot]

    def update_channel_names(self, new_channel_names):
        self.channel_names = list(new_channel_names)

    def get_all_biomarkers_for_channel(self, channel, t_start=0, t_end=sys.maxsize):
        return self.backend.event_features.get_biomarkers_for_channel(channel, t_start, t_end)

    def _resolve_event_color(self, artifacts, spikes, ehfos, index, fallback_color):
        try:
            if artifacts[index] == -1:
                return self.color_dict["non_spike"]
            if int(artifacts[index]) < 1:
                return self.color_dict["artifact"]
            if spikes[index]:
                return self.color_dict["spike"]
            if ehfos[index]:
                return self.color_dict["ehfo"]
            return self.color_dict["non_spike"]
        except Exception:
            return fallback_color

    def _build_channel_overlay(self, event_features, channel, group_color, is_active, t_start=0, t_end=sys.maxsize):
        if event_features is None:
            return np.array([]), np.array([]), []
        biomarker_payload = event_features.get_biomarkers_for_channel(channel, t_start, t_end)
        if len(biomarker_payload) == 5:
            starts, ends, artifacts, spikes, ehfos = biomarker_payload
        elif len(biomarker_payload) == 4:
            starts, ends, artifacts, spikes = biomarker_payload
            ehfos = np.zeros(len(starts), dtype=bool)
        else:
            return np.array([]), np.array([]), []

        colors = []
        for index in range(len(starts)):
            colors.append(
                self._resolve_event_color(artifacts, spikes, ehfos, index, group_color)
                if is_active
                else group_color
            )
        starts_in_time = np.asarray(starts, dtype=np.float64) / float(self.sample_freq)
        ends_in_time = np.asarray(np.minimum(ends, max(self.n_samples - 1, 0)), dtype=np.float64) / float(self.sample_freq)
        return starts_in_time, ends_in_time, colors

    def get_all_biomarkers_for_channel_and_color(self, channel, t_start=0, t_end=sys.maxsize):
        visible_runs = []
        if callable(self.overlay_run_provider):
            try:
                visible_runs = list(self.overlay_run_provider() or [])
            except Exception:
                visible_runs = []

        session = getattr(self.backend, "analysis_session", None)
        if not visible_runs and session is not None and hasattr(session, "get_visible_runs"):
            visible_runs = session.get_visible_runs()

        if not visible_runs:
            return self._build_channel_overlay(
                getattr(self.backend, "event_features", None),
                channel,
                self.color_dict["non_spike"],
                True,
                t_start,
                t_end,
            )

        active_run_id = getattr(session, "active_run_id", None) if session is not None else None
        starts_in_time = []
        ends_in_time = []
        colors = []
        for order_index, run in enumerate(visible_runs):
            group_color = self.RUN_OVERLAY_COLORS[order_index % len(self.RUN_OVERLAY_COLORS)]
            run_starts, run_ends, run_colors = self._build_channel_overlay(
                getattr(run, "event_features", None),
                channel,
                group_color,
                run.run_id == active_run_id,
                t_start,
                t_end,
            )
            if len(run_starts) == 0:
                continue
            starts_in_time.extend(run_starts.tolist())
            ends_in_time.extend(run_ends.tolist())
            colors.extend(run_colors)
        return np.array(starts_in_time), np.array(ends_in_time), colors
