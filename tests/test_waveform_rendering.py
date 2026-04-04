import numpy as np
import pyqtgraph as pg
import pytest
from PyQt5 import QtCore

from src.controllers.main_waveform_plot_controller import MainWaveformPlotController
from src.hfo_feature import HFO_Feature
from src.models.main_waveform_plot_model import MainWaveformPlotModel
from src.models.mini_plot_model import MiniPlotModel
from src.views.shared_plot_handle import SharedPlotHandle


class FakeEventFeatures:
    def get_biomarkers_for_channel(self, channel_name, min_start=None, max_end=None):
        starts = np.array([50, 2000, 4200], dtype=int)
        ends = np.array([70, 2040, 4260], dtype=int)
        artifacts = np.ones_like(starts)
        spikes = np.zeros_like(starts)
        ehfos = np.zeros_like(starts)
        if min_start is not None and max_end is not None:
            mask = (starts >= min_start) & (ends <= max_end)
            starts = starts[mask]
            ends = ends[mask]
            artifacts = artifacts[mask]
            spikes = spikes[mask]
            ehfos = ehfos[mask]
        return starts, ends, artifacts, spikes, ehfos


class FakeBackend:
    def __init__(self, n_channels=4, n_samples=200_000, sample_freq=1000.0, channel_names=None):
        self.sample_freq = sample_freq
        if channel_names is None:
            channel_names = [f"A{i}Ref" for i in range(1, n_channels + 1)]
        self._channel_names = np.array(channel_names)
        n_channels = len(self._channel_names)
        times = np.arange(n_samples) / sample_freq
        self._eeg = np.vstack(
            [
                np.sin(2 * np.pi * (idx + 1) * 4 * times) * max(0.25, 1 - idx * 0.1)
                for idx in range(n_channels)
            ]
        )
        self.event_features = FakeEventFeatures()

    def get_eeg_data(self, start=None, end=None, filtered=False):
        if start is None and end is None:
            return self._eeg, self._channel_names
        return self._eeg[:, start:end], self._channel_names

    def get_edf_info(self):
        return {"sfreq": self.sample_freq}


class PartialOverlapBackend(FakeBackend):
    def __init__(self):
        super().__init__(n_channels=1, n_samples=10_000, sample_freq=1000.0, channel_names=["A1Ref"])
        self.event_features = HFO_Feature(
            np.array(["A1Ref"]),
            np.array([[1950, 2050]]),
            sample_freq=self.sample_freq,
        )
        self.event_features.update_pred(np.array([1]), np.array([0]), np.array([0]))


def _process_events(qapp, cycles=6):
    for _ in range(cycles):
        qapp.processEvents()


class _FakeSceneClickEvent:
    def __init__(self, scene_pos):
        self._scene_pos = scene_pos
        self.accepted = False

    def button(self):
        return QtCore.Qt.LeftButton

    def scenePos(self):
        return self._scene_pos

    def accept(self):
        self.accepted = True


def test_main_waveform_plot_model_downsamples_long_windows():
    backend = FakeBackend()
    model = MainWaveformPlotModel(backend)

    model.init_eeg_data()
    model.set_render_width_pixels(600)
    model.set_time_window(50)
    model.set_current_time_window(0)

    eeg_data, y_100_length, y_scale_length, offset_value = model.get_all_current_eeg_data_to_display()
    time_axis = model.get_current_time_window()

    assert not hasattr(model, "time")
    assert model.get_total_time() > 0
    assert eeg_data.shape[1] == len(time_axis)
    assert eeg_data.shape[1] <= model._target_render_points() + 1
    assert y_100_length in (50, 100)
    assert y_scale_length > 0
    assert offset_value == 6


def test_main_waveform_plot_model_clips_partial_overlap_event_segments():
    backend = PartialOverlapBackend()
    model = MainWaveformPlotModel(backend)

    model.init_eeg_data()
    model.set_n_channels_to_plot(1)
    model.set_time_window(0.1)
    model.set_current_time_window(2.0)
    model.get_all_current_eeg_data_to_display()

    overlay_groups = model.get_all_biomarkers_for_all_current_channels_and_color("A1Ref")

    assert len(overlay_groups) == 1
    assert overlay_groups[0]["starts"].tolist() == [1950]
    assert overlay_groups[0]["ends"].tolist() == [2050]

    segment_x, segment_y = model.get_event_display_segment(0, 1950, 2050)

    assert len(segment_x) == len(segment_y)
    assert len(segment_x) > 0
    assert segment_x[0] == pytest.approx(2.0)
    assert segment_x[-1] < 2.1


def test_mini_plot_model_tracks_total_time_without_full_time_array():
    backend = FakeBackend()
    model = MiniPlotModel(backend)

    model.init_eeg_data()
    starts_in_time, ends_in_time, _ = model.get_all_biomarkers_for_channel_and_color("A1Ref")

    assert not hasattr(model, "time")
    assert model.total_time > 0
    assert np.all(ends_in_time >= starts_in_time)


def test_main_waveform_gutter_is_reserved_for_longest_selected_channel_name(qapp):
    channel_names = [
        "POL ST11",
        "POL ST12",
        "POL ST13",
        "POL ST14",
        "POL ST15",
        "EEG IO1-Ref",
        "EEG IO2-Ref",
        "EEG SO1-Ref",
    ]
    backend = FakeBackend(channel_names=channel_names, n_samples=10_000)
    graphics_widget = pg.GraphicsLayoutWidget()
    gutter_item = graphics_widget.addPlot(row=0, col=0)
    main_item = graphics_widget.addPlot(row=0, col=1)
    graphics_widget.resize(720, 480)
    graphics_widget.show()
    _process_events(qapp)

    plot_widget = SharedPlotHandle(graphics_widget, main_item)
    gutter_widget = SharedPlotHandle(graphics_widget, gutter_item)

    controller = MainWaveformPlotController(plot_widget, backend, channel_gutter_widget=gutter_widget)
    try:
        controller.init_eeg_data()
        controller.init_waveform_display()
        controller.set_current_time_window(0)
        controller.set_n_channels_to_plot(4)

        _, _, _, offset_value = controller.plot_all_current_channels_for_window()
        controller.draw_channel_names(offset_value)
        _process_events(qapp)

        expected_width = controller.view.calculate_left_axis_width(channel_names)
        first_page_width = controller.view.calculate_left_axis_width(channel_names[:4])
        main_height = controller.view.plot_widget.getPlotItem().vb.sceneBoundingRect().height()
        gutter_height = controller.view.channel_gutter_widget.getPlotItem().vb.sceneBoundingRect().height()

        assert expected_width > first_page_width
        assert controller.uses_channel_gutter() is True
        assert controller.get_left_axis_width() == expected_width
        assert abs(main_height - gutter_height) < 0.1
        assert [item.toolTip() for item in controller.view._channel_label_items] == channel_names[:4]

        controller.clear()
        controller.set_first_channel_to_plot(4)
        _, _, _, offset_value = controller.plot_all_current_channels_for_window()
        controller.draw_channel_names(offset_value)
        _process_events(qapp)

        assert controller.get_left_axis_width() == expected_width
        assert [item.toolTip() for item in controller.view._channel_label_items] == channel_names[4:8]
    finally:
        graphics_widget.close()
        _process_events(qapp)


def test_main_waveform_gutter_elides_long_labels_and_clicks_only_highlight_rows(qapp):
    long_channel_name = "EEG VERY LONG TEMPORAL CHANNEL NAME THAT SHOULD ELIDE IN THE GUTTER"
    channel_names = [long_channel_name, "POL ST11", "POL ST12", "POL ST13"]
    backend = FakeBackend(channel_names=channel_names, n_samples=10_000)
    graphics_widget = pg.GraphicsLayoutWidget()
    gutter_item = graphics_widget.addPlot(row=0, col=0)
    main_item = graphics_widget.addPlot(row=0, col=1)
    graphics_widget.resize(720, 480)
    graphics_widget.show()
    _process_events(qapp)

    plot_widget = SharedPlotHandle(graphics_widget, main_item)
    gutter_widget = SharedPlotHandle(graphics_widget, gutter_item)

    controller = MainWaveformPlotController(plot_widget, backend, channel_gutter_widget=gutter_widget)
    try:
        controller.init_eeg_data()
        controller.init_waveform_display()
        controller.set_current_time_window(0)
        controller.set_n_channels_to_plot(4)

        _, _, _, offset_value = controller.plot_all_current_channels_for_window()
        controller.draw_channel_names(offset_value)
        _process_events(qapp)

        first_label = controller.view._channel_label_items[0]
        assert first_label.toPlainText() != long_channel_name
        assert first_label.toolTip() == long_channel_name

        channels_before = list(controller.model.channels_to_plot)
        activated = controller.view._activate_channel_for_y(controller.view._visible_channel_locations[0])
        assert activated == long_channel_name
        assert controller.view.highlighted_channel_name == long_channel_name
        assert list(controller.model.channels_to_plot) == channels_before
        assert controller.view._main_row_highlight is not None
        assert controller.view._main_row_highlight.isVisible() is True
        assert controller.view._gutter_row_highlight is not None
        assert controller.view._gutter_row_highlight.isVisible() is True
    finally:
        graphics_widget.close()
        _process_events(qapp)


def test_main_waveform_gutter_uses_channel_presentation_labels_and_tooltips(qapp):
    channel_names = ["A1Ref#-#A2Ref", "A2Ref#-#A3Ref"]
    backend = FakeBackend(channel_names=channel_names, n_samples=10_000)
    graphics_widget = pg.GraphicsLayoutWidget()
    gutter_item = graphics_widget.addPlot(row=0, col=0)
    main_item = graphics_widget.addPlot(row=0, col=1)
    graphics_widget.resize(720, 480)
    graphics_widget.show()
    _process_events(qapp)

    plot_widget = SharedPlotHandle(graphics_widget, main_item)
    gutter_widget = SharedPlotHandle(graphics_widget, gutter_item)

    controller = MainWaveformPlotController(plot_widget, backend, channel_gutter_widget=gutter_widget)
    try:
        controller.init_eeg_data()
        controller.init_waveform_display()
        controller.set_channel_presentation(
            {
                "A1Ref#-#A2Ref": {
                    "display_label": "A1-A2",
                    "tooltip": "A1-A2\nSource: A1Ref - A2Ref",
                },
                "A2Ref#-#A3Ref": {
                    "display_label": "A2-A3",
                    "tooltip": "A2-A3\nSource: A2Ref - A3Ref",
                },
            }
        )
        controller.set_current_time_window(0)
        controller.set_n_channels_to_plot(2)

        _, _, _, offset_value = controller.plot_all_current_channels_for_window()
        controller.draw_channel_names(offset_value)
        _process_events(qapp)

        rendered_labels = [item.toPlainText() for item in controller.view._channel_label_items]
        assert rendered_labels[0] == "A1-A2"
        assert rendered_labels[1] != "A2Ref#-#A3Ref"
        assert [item.toolTip() for item in controller.view._channel_label_items] == [
            "A1-A2\nSource: A1Ref - A2Ref",
            "A2-A3\nSource: A2Ref - A3Ref",
        ]
    finally:
        graphics_widget.close()
        _process_events(qapp)


def test_main_waveform_hover_highlights_rows_without_committing_selection(qapp):
    channel_names = ["A1Ref", "A2Ref", "A3Ref"]
    backend = FakeBackend(channel_names=channel_names, n_samples=10_000)
    graphics_widget = pg.GraphicsLayoutWidget()
    gutter_item = graphics_widget.addPlot(row=0, col=0)
    main_item = graphics_widget.addPlot(row=0, col=1)
    graphics_widget.resize(720, 480)
    graphics_widget.show()
    _process_events(qapp)

    plot_widget = SharedPlotHandle(graphics_widget, main_item)
    gutter_widget = SharedPlotHandle(graphics_widget, gutter_item)

    controller = MainWaveformPlotController(plot_widget, backend, channel_gutter_widget=gutter_widget)
    try:
        controller.init_eeg_data()
        controller.init_waveform_display()
        controller.set_current_time_window(0)
        controller.set_n_channels_to_plot(3)

        _, _, _, offset_value = controller.plot_all_current_channels_for_window()
        controller.draw_channel_names(offset_value)
        _process_events(qapp)

        hover_y = controller.view._visible_channel_locations[1]
        controller.view._set_hovered_channel(controller.view._resolve_channel_name_for_y(hover_y))

        assert controller.view._hovered_channel_name == channel_names[1]
        assert controller.view.highlighted_channel_name is None
        assert controller.get_highlighted_channel() is None
        assert controller.view._main_row_highlight is not None
        assert controller.view._main_row_highlight.isVisible() is True
    finally:
        graphics_widget.close()
        _process_events(qapp)


def test_main_waveform_scale_bar_uses_inset_capsule_label(qapp):
    backend = FakeBackend(channel_names=["A1Ref", "A2Ref"], n_samples=10_000)
    graphics_widget = pg.GraphicsLayoutWidget()
    main_item = graphics_widget.addPlot(row=0, col=0)
    graphics_widget.resize(720, 480)
    graphics_widget.show()
    _process_events(qapp)

    plot_widget = SharedPlotHandle(graphics_widget, main_item)
    controller = MainWaveformPlotController(plot_widget, backend)
    try:
        controller.init_eeg_data()
        controller.init_waveform_display()
        controller.set_current_time_window(0)
        controller.set_n_channels_to_plot(2)

        eeg_data_to_display, y_100_length, y_scale_length, offset_value = controller.plot_all_current_channels_for_window()
        controller.draw_channel_names(offset_value)
        controller.draw_scale_bar(eeg_data_to_display, offset_value, y_100_length, y_scale_length)
        _process_events(qapp)

        assert len(controller.view._scale_bar_items) == 3
        assert controller.view._scale_text_item is not None
        assert controller.view._scale_text_item.toPlainText().strip() == f"{y_100_length} μV"

        vertical_x, vertical_y = controller.view._scale_bar_items[0].getData()
        bottom_cap_x, bottom_cap_y = controller.view._scale_bar_items[1].getData()
        top_cap_x, top_cap_y = controller.view._scale_bar_items[2].getData()

        assert vertical_x[0] == pytest.approx(vertical_x[1])
        assert vertical_y[1] - vertical_y[0] == pytest.approx(y_scale_length)
        assert bottom_cap_y[0] == pytest.approx(bottom_cap_y[1])
        assert top_cap_y[0] == pytest.approx(top_cap_y[1])
        assert bottom_cap_x[0] < vertical_x[0] < bottom_cap_x[1]
        assert top_cap_x[0] < vertical_x[0] < top_cap_x[1]
        assert controller.view._scale_text_item.pos().x() < vertical_x[0]
    finally:
        graphics_widget.close()
        _process_events(qapp)


def test_main_waveform_click_selects_channel_without_measurement_mode(qapp):
    channel_names = ["A1Ref", "A2Ref", "A3Ref"]
    backend = FakeBackend(channel_names=channel_names, n_samples=10_000)
    graphics_widget = pg.GraphicsLayoutWidget()
    gutter_item = graphics_widget.addPlot(row=0, col=0)
    main_item = graphics_widget.addPlot(row=0, col=1)
    graphics_widget.resize(720, 480)
    graphics_widget.show()
    _process_events(qapp)

    plot_widget = SharedPlotHandle(graphics_widget, main_item)
    gutter_widget = SharedPlotHandle(graphics_widget, gutter_item)

    controller = MainWaveformPlotController(plot_widget, backend, channel_gutter_widget=gutter_widget)
    selected_channels = []
    controller.view.channel_gutter_clicked.connect(selected_channels.append)
    try:
        controller.init_eeg_data()
        controller.init_waveform_display()
        controller.set_current_time_window(0)
        controller.set_n_channels_to_plot(3)

        _, _, _, offset_value = controller.plot_all_current_channels_for_window()
        controller.draw_channel_names(offset_value)
        _process_events(qapp)

        target_time = float(controller.get_current_time_window()[0])
        target_y = float(controller.view._visible_channel_locations[1])
        scene_pos = controller.view.plot_widget.getPlotItem().vb.mapViewToScene(QtCore.QPointF(target_time, target_y))
        event = _FakeSceneClickEvent(scene_pos)

        controller.view._on_plot_scene_clicked(event)

        assert controller.is_measurement_enabled() is False
        assert event.accepted is True
        assert controller.get_highlighted_channel() == channel_names[1]
        assert selected_channels == [channel_names[1]]
    finally:
        graphics_widget.close()
        _process_events(qapp)
