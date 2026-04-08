import json
import numpy as np
from pathlib import Path
from PyQt5 import uic
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy
from src.ui.waveform_toolbar_icons import get_waveform_toolbar_icon
from src.ui.ui_tokens import UI_DENSITY, resolve_ui_density
from src.utils.utils_gui import *

ROOT_DIR = Path(__file__).parent.parent.parent


class MainWindowView(QObject):
    def __init__(self, window):
        super(MainWindowView, self).__init__()
        self.window = window
        self._placeholder_target = None
        self._inspector_section_roots = {}
        self._inspector_option_counter = 0
        self._inspector_view_state = self._load_inspector_view_state()
        self.ui_density = UI_DENSITY
        self._button_feedback_duration_ms = 220
        # self._init_plot_widget(plot_widget)

    def _refresh_ui_density(self):
        screen = None
        if hasattr(self.window, "windowHandle") and self.window.windowHandle() is not None:
            screen = self.window.windowHandle().screen()
        if screen is None and hasattr(self.window, "screen"):
            screen = self.window.screen()
        self.ui_density = resolve_ui_density(screen)

    def _load_inspector_view_state(self):
        settings = QtCore.QSettings("PyBrain", "PyBrain")
        raw = settings.value("right_panel/section_visibility", "{}")
        if isinstance(raw, dict):
            return raw
        if raw in (None, ""):
            return {}
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return {}

    def _save_inspector_view_state(self):
        settings = QtCore.QSettings("PyBrain", "PyBrain")
        settings.setValue("right_panel/section_visibility", json.dumps(self._inspector_view_state))

    def _get_section_view_state(self, section_id):
        state = self._inspector_view_state.get(section_id, {})
        return {
            "mode": state.get("mode", "standard"),
            "custom": dict(state.get("custom", {})),
        }

    def _set_section_view_state(self, section_id, state):
        self._inspector_view_state[section_id] = {
            "mode": state.get("mode", "standard"),
            "custom": dict(state.get("custom", {})),
        }
        self._save_inspector_view_state()

    def _normalize_option_id(self, text):
        normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text or "option"))
        normalized = "_".join(part for part in normalized.split("_") if part)
        return normalized or "option"

    def _compact_label_width(self):
        return self.ui_density.compact_label_width

    def _compact_unit_width(self):
        return self.ui_density.compact_unit_width

    def _apply_compact_input_height(self, widget):
        if widget is None:
            return
        target_height = self.ui_density.compact_input_height
        widget.setFixedHeight(target_height)

    def _compact_label_text(self, text):
        label_map = {
            "Low cutoff": "Low cut.",
            "High cutoff": "High cut.",
            "Pass ripple": "Ripple",
            "Stop atten.": "Stop attn.",
            "RMS window": "RMS win.",
            "Min window": "Min win.",
            "Min gap": "Min gap",
            "Min oscill.": "Min osc.",
            "Peak thresh.": "Peak thr.",
            "RMS thresh.": "RMS thr.",
            "Epoch length": "Epoch len.",
            "Epoch time": "Epoch",
            "CHF percent": "CHF %",
            "Baseline win.": "Base win.",
            "Baseline shift": "Base shift",
            "Baseline thres.": "Base thr.",
            "Baseline min": "Base min",
            "Sample freq.": "Fs",
            "SD thresh.": "SD thr.",
            "Min distance": "Min dist.",
            "Rel power": "Rel pow.",
            "Correlation": "Corr.",
            "Spindle band": "Spin band",
            "Broad band": "Broad bd.",
            "Duration": "Dur.",
        }
        return label_map.get(str(text or ""), str(text or ""))

    def _ensure_button_feedback_timer(self, button):
        timer = getattr(button, "_click_feedback_timer", None)
        if timer is None:
            timer = QtCore.QTimer(button)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda current=button: self._set_button_click_feedback(current, False))
            button._click_feedback_timer = timer
        return timer

    def _set_button_click_feedback(self, button, active):
        if button is None:
            return
        set_dynamic_property(button, "clickFeedback", bool(active))

    def _begin_button_click_feedback(self, button):
        if button is None or not button.isEnabled():
            return
        timer = self._ensure_button_feedback_timer(button)
        timer.stop()
        self._set_button_click_feedback(button, True)

    def _arm_button_click_feedback(self, button):
        if button is None:
            return
        timer = self._ensure_button_feedback_timer(button)
        self._set_button_click_feedback(button, True)
        timer.start(self._button_feedback_duration_ms)

    def _install_button_click_feedback(self, root):
        if root is None:
            return
        buttons = []
        if isinstance(root, (QtWidgets.QPushButton, QtWidgets.QToolButton)):
            buttons.append(root)
        buttons.extend(root.findChildren(QtWidgets.QPushButton))
        buttons.extend(root.findChildren(QtWidgets.QToolButton))

        seen = set()
        for button in buttons:
            button_id = id(button)
            if button_id in seen or button.property("clickFeedbackInstalled"):
                continue
            seen.add(button_id)
            button.setProperty("clickFeedbackInstalled", True)
            button.setProperty("clickFeedback", False)
            button.pressed.connect(lambda current=button: self._begin_button_click_feedback(current))
            button.clicked.connect(lambda _checked=False, current=button: self._arm_button_click_feedback(current))

    def _configure_compact_form_grid(self, grid, *, action_column=False):
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)
        if action_column:
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(2, 0)
            return
        grid.setColumnMinimumWidth(0, self._compact_label_width())
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(2, self._compact_unit_width())
        grid.setColumnMinimumWidth(3, self._compact_label_width())
        grid.setColumnStretch(4, 1)
        grid.setColumnMinimumWidth(5, self._compact_unit_width())

    def _add_compact_field_to_grid(self, grid, row, side, field_widget):
        column = 0 if side == 0 else 3
        grid.addWidget(field_widget, row, column, 1, 3)

    def _create_compact_control_row(self, label_text, input_widget, *, action_widgets=None, label_widget=None, option_id=None, density="standard"):
        frame = QFrame()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        raw_label_text = label_widget.text() if label_widget is not None and label_widget.text() else label_text
        label = label_widget or QLabel(raw_label_text)
        label.setProperty("fieldLabel", True)
        label.setText(self._compact_label_text(raw_label_text))
        label.setFixedWidth(self._compact_label_width())
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        layout.addWidget(label)

        self._apply_compact_input_height(input_widget)
        input_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(input_widget, 1)
        input_widget.show()

        if action_widgets:
            action_frame = QFrame(frame)
            action_layout = QHBoxLayout(action_frame)
            action_layout.setContentsMargins(0, 0, 0, 0)
            action_layout.setSpacing(3)
            for widget in action_widgets:
                action_layout.addWidget(widget)
            layout.addWidget(action_frame, 0, Qt.AlignRight)

        return self._mark_section_option(frame, option_id or self._normalize_option_id(raw_label_text), raw_label_text, density)

    def _mark_section_option(self, widget, option_id, label_text, density="standard"):
        self._inspector_option_counter += 1
        widget.setProperty("inspectorOptionId", option_id)
        widget.setProperty("inspectorOptionLabel", label_text)
        widget.setProperty("inspectorOptionDensity", density)
        widget.setProperty("inspectorOptionOrder", self._inspector_option_counter)
        return widget

    def _waveform_spinbox_slot_width(self, widget, fallback_width):
        if widget is None:
            return int(fallback_width)
        try:
            hint_width = max(widget.sizeHint().width(), widget.minimumSizeHint().width())
        except Exception:
            hint_width = int(fallback_width)
        return max(int(fallback_width), int(hint_width) + 4)

    def _collect_section_option_specs(self, section_id, *, visible_only=False):
        root = self._inspector_section_roots.get(section_id)
        if root is None:
            return []

        candidates = []
        if root.property("inspectorOptionId"):
            candidates.append(root)
        candidates.extend(root.findChildren(QWidget))

        specs = {}
        for widget in candidates:
            option_id = widget.property("inspectorOptionId")
            if not option_id:
                continue
            if visible_only and not widget.isVisibleTo(root):
                continue
            spec = specs.setdefault(
                option_id,
                {
                    "id": option_id,
                    "label": widget.property("inspectorOptionLabel") or option_id,
                    "density": widget.property("inspectorOptionDensity") or "standard",
                    "order": widget.property("inspectorOptionOrder") or 0,
                    "widgets": [],
                },
            )
            spec["widgets"].append(widget)
            spec["order"] = min(spec["order"], widget.property("inspectorOptionOrder") or spec["order"])

        ordered = sorted(specs.values(), key=lambda item: item["order"])
        if visible_only and not ordered:
            return self._collect_section_option_specs(section_id, visible_only=False)
        return ordered

    def _collect_menu_option_specs(self, section_id):
        if section_id != "DETECTION":
            return self._collect_section_option_specs(section_id, visible_only=False)

        current_tab = getattr(self.window, "detector_subtabs", None)
        current_widget = current_tab.currentWidget() if current_tab is not None else None
        if current_widget is None:
            return self._collect_section_option_specs(section_id, visible_only=False)

        candidates = []
        if current_widget.property("inspectorOptionId"):
            candidates.append(current_widget)
        candidates.extend(current_widget.findChildren(QWidget))

        specs = {}
        for widget in candidates:
            option_id = widget.property("inspectorOptionId")
            if not option_id:
                continue
            spec = specs.setdefault(
                option_id,
                {
                    "id": option_id,
                    "label": widget.property("inspectorOptionLabel") or option_id,
                    "density": widget.property("inspectorOptionDensity") or "standard",
                    "order": widget.property("inspectorOptionOrder") or 0,
                    "widgets": [],
                },
            )
            spec["widgets"].append(widget)
            spec["order"] = min(spec["order"], widget.property("inspectorOptionOrder") or spec["order"])
        return sorted(specs.values(), key=lambda item: item["order"])

    def _density_visible_for_mode(self, density, mode):
        density_rank = {"essential": 0, "standard": 1, "advanced": 2}
        mode_rank = {"essential": 0, "standard": 1, "all": 2}
        return density_rank.get(density, 1) <= mode_rank.get(mode, 1)

    def _apply_section_visibility(self, section_id):
        state = self._get_section_view_state(section_id)
        specs = self._collect_section_option_specs(section_id, visible_only=False)
        for spec in specs:
            visible = self._density_visible_for_mode(spec["density"], state["mode"])
            if spec["id"] in state["custom"]:
                visible = bool(state["custom"][spec["id"]])
            for widget in spec["widgets"]:
                widget.setVisible(visible)

    def _apply_all_section_visibility(self):
        for section_id in ("RUN_ACTIONS", "SIGNAL", "DETECTION", "CLASSIFICATION"):
            self._apply_section_visibility(section_id)

    def _set_section_mode(self, section_id, mode):
        state = self._get_section_view_state(section_id)
        state["mode"] = mode
        state["custom"] = {}
        self._set_section_view_state(section_id, state)
        self._apply_section_visibility(section_id)

    def _set_section_option_visibility(self, section_id, option_id, visible):
        state = self._get_section_view_state(section_id)
        state["custom"][option_id] = bool(visible)
        self._set_section_view_state(section_id, state)
        self._apply_section_visibility(section_id)

    def _reset_section_visibility(self, section_id):
        self._set_section_view_state(section_id, {"mode": "standard", "custom": {}})
        self._apply_section_visibility(section_id)

    def _create_section_settings_button(self, section_id, parent):
        button = QToolButton(parent)
        button_label = "Run" if section_id == "RUN_ACTIONS" else "More"
        tooltip_label_map = {
            "RUN_ACTIONS": "run actions",
            "SIGNAL": "signal",
            "DETECTION": "detection",
            "CLASSIFICATION": "classification",
        }
        button.setText(button_label)
        button.setToolTip(f"More options for the {tooltip_label_map.get(section_id, 'section')} section")
        button.setProperty("inspectorMenu", True)
        button.setAutoRaise(False)
        button.clicked.connect(lambda _checked=False, sid=section_id, btn=button: self._show_section_settings_menu(sid, btn))
        return button

    def _show_section_settings_menu(self, section_id, button):
        menu = QMenu(button)
        state = self._get_section_view_state(section_id)
        current_mode = state["mode"]

        mode_actions = [
            ("essential", "Essential"),
            ("standard", "Standard"),
            ("all", "All"),
        ]
        for mode_id, label in mode_actions:
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(current_mode == mode_id)
            action.triggered.connect(lambda _checked=False, sid=section_id, mid=mode_id: self._set_section_mode(sid, mid))

        specs = self._collect_menu_option_specs(section_id)
        if specs:
            menu.addSeparator()
            for spec in specs:
                visible = self._density_visible_for_mode(spec["density"], current_mode)
                if spec["id"] in state["custom"]:
                    visible = bool(state["custom"][spec["id"]])
                action = menu.addAction(spec["label"])
                action.setCheckable(True)
                action.setChecked(visible)
                action.triggered.connect(
                    lambda checked, sid=section_id, oid=spec["id"]: self._set_section_option_visibility(sid, oid, checked)
                )

        menu.addSeparator()
        reset_action = menu.addAction("Reset")
        reset_action.triggered.connect(lambda _checked=False, sid=section_id: self._reset_section_visibility(sid))
        menu.popup(button.mapToGlobal(QtCore.QPoint(0, button.height())))

    def init_general_window(self):
        self.window.ui = uic.loadUi(os.path.join(ROOT_DIR, 'src/ui/main_window.ui'), self.window)
        self.window.setWindowIcon(QtGui.QIcon(os.path.join(ROOT_DIR, 'src/ui/images/icon1.png'))) 
        self.window.setWindowTitle("PyBrain")
        self._refresh_ui_density()
        screen = self.window.screen() or QtWidgets.QApplication.primaryScreen()
        rect = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1440, 900)
        width = max(1100, min(int(rect.width() * 0.9), 1360))
        height = max(720, min(int(rect.height() * 0.88), 860))
        self.window.resize(width, height)
        self.window.setMinimumSize(1024, 680)

        self._configure_hub_model_banner()

        self.window.threadpool = QThreadPool()
        self.window.replace_last_line = False
        self._install_report_actions()
        self._apply_window_theme()
        self._retitle_workflow_sections()
        self._configure_native_toolbars()
        self._build_native_window_shell()
        self._refine_waveform_control_strip()
        self._move_runtime_controls_to_prepare_panel()
        self._enhance_analysis_setup_tabs()
        self._streamline_prepare_tab()
        self._build_single_page_inspector()
        self._install_button_click_feedback(getattr(self.window, "inspector_container", None))
        self._install_button_click_feedback(getattr(self.window, "run_stats_dialog", None))
        apply_compact_button_heights(self.window, self.ui_density)
        apply_compact_input_heights(self.window, self.ui_density)
        QtCore.QTimer.singleShot(0, self._apply_default_dock_sizes)
        self.set_workspace_state(False, self.window.combo_box_biomarker.currentText())

    def _apply_window_theme(self):
        button_min_height = max(16, self.ui_density.compact_button_height - 4)
        tool_min_height = max(14, self.ui_density.compact_tool_height)
        waveform_header_title_font = max(12, self.ui_density.waveform_header_title_font)
        waveform_header_meta_font = max(9, self.ui_density.waveform_header_meta_font)
        waveform_header_meta_strong_font = max(10, waveform_header_meta_font + 1)
        legend_label_font = max(10, self.ui_density.base_font)
        popup_stylesheet = build_popup_chrome_stylesheet(
            self.ui_density,
            control_radius=self.ui_density.compact_radius,
            popup_radius=self.ui_density.group_radius,
            item_radius=self.ui_density.compact_radius,
            drop_width=18,
            item_height=self.ui_density.compact_input_height,
            font_size=self.ui_density.button_font,
            input_padding_right=24,
            drop_background="#f3f6f9",
        )
        checkbox_stylesheet = build_checkbox_styles(
            size=16,
            radius=5,
            spacing=6,
            label_color="#33495b",
            label_disabled_color="#94a2af",
            border_color="#b9c5cf",
            hover_border_color="#8ea0b1",
            background="#ffffff",
            hover_background="#f5f8fb",
            accent="#355c72",
            accent_hover="#2d4e61",
            disabled_border="#d7dee5",
            disabled_background="#f5f7f9",
            disabled_accent="#b6c1cc",
        )
        inspector_checkbox_stylesheet = build_checkbox_styles(
            prefix="#InspectorSinglePage ",
            size=14,
            radius=4,
            spacing=4,
            label_color="#334155",
            label_disabled_color="#93a0ad",
            border_color="#cbd5df",
            hover_border_color="#9aa9b8",
            background="#f8f9fa",
            hover_background="#ffffff",
            accent="#355c72",
            accent_hover="#2d4e61",
            disabled_border="#d7dee5",
            disabled_background="#f3f5f7",
            disabled_accent="#b9c4cf",
        )
        waveform_stepper_up_icon = (ROOT_DIR / "src" / "ui" / "images" / "chevron_up.svg").resolve().as_posix()
        waveform_stepper_down_icon = (ROOT_DIR / "src" / "ui" / "images" / "chevron_down.svg").resolve().as_posix()
        waveform_stepper_stylesheet = f"""
            QAbstractSpinBox[waveformField="true"][waveformStepper="true"] {{
                padding-right: 22px;
            }}
            QAbstractSpinBox[waveformField="true"][waveformStepper="true"]::up-button {{
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 18px;
                border-left: 1px solid #d7e0e7;
                border-top-right-radius: 7px;
                background: #f4f8fb;
            }}
            QAbstractSpinBox[waveformField="true"][waveformStepper="true"]::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 18px;
                border-left: 1px solid #d7e0e7;
                border-top: 1px solid #d7e0e7;
                border-bottom-right-radius: 7px;
                background: #f4f8fb;
            }}
            QAbstractSpinBox[waveformField="true"][waveformStepper="true"]::up-button:hover,
            QAbstractSpinBox[waveformField="true"][waveformStepper="true"]::down-button:hover {{
                background: #edf4f9;
            }}
            QAbstractSpinBox[waveformField="true"][waveformStepper="true"]::up-button:pressed,
            QAbstractSpinBox[waveformField="true"][waveformStepper="true"]::down-button:pressed {{
                background: #e4edf4;
            }}
            QAbstractSpinBox[waveformField="true"][waveformStepper="true"]::up-arrow {{
                image: url("{waveform_stepper_up_icon}");
                width: 10px;
                height: 10px;
            }}
            QAbstractSpinBox[waveformField="true"][waveformStepper="true"]::down-arrow {{
                image: url("{waveform_stepper_down_icon}");
                width: 10px;
                height: 10px;
            }}
        """
        self.window.setStyleSheet("""
            QMainWindow {
                background: #f3f4f6;
            }
            QWidget {
                color: #243746;
                font-size: 10px;
            }
            QMainWindow::separator {
                width: 1px;
                height: 1px;
                background: #d7dce2;
            }
            QToolBar {
                background: #f7f8fa;
                border: none;
                border-bottom: 1px solid #d8dde3;
                spacing: 3px;
                padding: 2px 6px;
            }
            QStatusBar {
                background: #eef2f5;
                color: #4a6174;
                border-top: 1px solid #d8dde3;
            }
            QStatusBar::item {
                border: none;
            }
            QToolBar QToolButton {
                background: transparent;
                color: #243746;
                border: 1px solid transparent;
                border-radius: 5px;
                padding: 1px 5px;
                font-size: 9px;
                margin: 0 1px;
            }
            QToolBar QToolButton:hover {
                background: #edf1f5;
                border-color: #d8dde3;
            }
            QToolBar QToolButton:pressed {
                background: #e3e8ee;
            }
            QToolBar QToolButton:checked {
                background: #e8edf3;
                border-color: #c7d1da;
            }
            QDockWidget {
                color: #243746;
                font-weight: 600;
                background: #f4f6f8;
                border-left: 1px solid #dfe4e8;
            }
            QDockWidget::title {
                background: transparent;
                border: none;
                padding: 0px;
            }
            QGroupBox {
                border: 1px solid #e0e6ec;
                border-radius: 7px;
                margin-top: 6px;
                padding-top: 6px;
                background: #fbfcfd;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 2px;
                color: #314657;
            }
            QPushButton {
                background: #ffffff;
                color: #243746;
                border: 1px solid #cfd6dd;
                border-radius: 5px;
                padding: 1px 6px;
                font-size: 9px;
            }
            QPushButton:hover {
                background: #f7fafc;
                border-color: #bec9d3;
            }
            QPushButton:pressed {
                background: #edf3f7;
                border-color: #9eb0bf;
            }
            QPushButton:disabled {
                background: #f7f8fa;
                color: #9aa6b2;
                border: 1px solid #dde2e8;
            }
            QPushButton[workflowBusy="true"], QToolButton[workflowBusy="true"] {
                background: #58748a;
                color: #ffffff;
                border-color: #58748a;
                font-weight: 700;
            }
            QPushButton[workflowBusy="true"]:disabled, QToolButton[workflowBusy="true"]:disabled {
                background: #58748a;
                color: #ffffff;
                border-color: #58748a;
            }
            QToolButton {
                background: #ffffff;
                color: #243746;
                border: 1px solid #cfd6dd;
                border-radius: 5px;
                padding: 1px 6px;
                font-size: 9px;
            }
            QToolButton:hover {
                background: #f7fafc;
                border-color: #bec9d3;
            }
            QToolButton:pressed {
                background: #edf3f7;
                border-color: #9eb0bf;
            }
            QToolButton[inspectorMenu="true"] {
                min-height: 16px;
                padding: 0 5px;
                border-radius: 4px;
                font-size: 9px;
                color: #4b5563;
                background: #f8f9fa;
                border: 1px solid #d9dee3;
            }
            QToolButton[inspectorMenu="true"]:hover {
                background: #ffffff;
                border-color: #c6d0da;
            }
            QPushButton[inspectorPrimary="true"], QToolButton[inspectorPrimary="true"] {
                background: #355c72;
                color: #ffffff;
                border: 1px solid #355c72;
                font-weight: 700;
            }
            QPushButton[inspectorPrimary="true"]:hover, QToolButton[inspectorPrimary="true"]:hover {
                background: #2d4e61;
                border-color: #2d4e61;
            }
            QPushButton[inspectorPrimary="true"]:pressed, QToolButton[inspectorPrimary="true"]:pressed {
                background: #234356;
                border-color: #234356;
            }
            QPushButton[inspectorPrimary="true"]:disabled, QToolButton[inspectorPrimary="true"]:disabled {
                background: #d7dde2;
                color: #f7f8fa;
                border-color: #d7dde2;
            }
            QPushButton[inspectorSecondary="true"], QToolButton[inspectorSecondary="true"] {
                background: #fffdfa;
                color: #2f495b;
                border: 1px solid #cfc6ba;
                font-weight: 600;
            }
            QPushButton[inspectorSecondary="true"]:hover, QToolButton[inspectorSecondary="true"]:hover {
                background: #f4eee6;
                border-color: #c4b7a6;
            }
            QPushButton[inspectorSecondary="true"]:pressed, QToolButton[inspectorSecondary="true"]:pressed {
                background: #ece1d3;
                border-color: #b29f8b;
            }
            QToolButton[runLauncher="true"] {
                background: #f5f8fb;
                color: #31495a;
                border: 1px solid #d4dde5;
                border-radius: 4px;
                padding: 0 18px 0 8px;
                font-weight: 600;
            }
            QToolButton[runLauncher="true"]:hover {
                background: #ffffff;
                border-color: #bccad7;
            }
            QToolButton[runLauncher="true"]:pressed {
                background: #e8f0f6;
                border-color: #8ea2b3;
            }
            QToolButton[runLauncher="true"]:disabled {
                background: #f5f7f9;
                color: #9aa6b2;
                border-color: #dde3e8;
            }
            QToolButton[runLauncher="true"]::menu-indicator {
                subcontrol-origin: padding;
                subcontrol-position: center right;
            }
            QPushButton[clickFeedback="true"], QToolButton[clickFeedback="true"] {
                background: #e9f1f7;
                border-color: #7f9bb1;
            }
            QPushButton[inspectorPrimary="true"][clickFeedback="true"], QToolButton[inspectorPrimary="true"][clickFeedback="true"] {
                background: #1f4356;
                border-color: #1f4356;
                color: #ffffff;
            }
            QPushButton[inspectorSecondary="true"][clickFeedback="true"], QToolButton[inspectorSecondary="true"][clickFeedback="true"] {
                background: #e8ddd0;
                border-color: #9f8f7c;
                color: #2f495b;
            }
            QToolButton[waveformTool="true"] {
                min-height: 24px;
                min-width: 36px;
                padding-left: 4px;
                padding-right: 2px;
                border-radius: 7px;
                border: 1px solid #d7e0e7;
                background: #fbfcfd;
                color: #2a3f50;
                font-size: 9px;
                font-weight: 600;
            }
            QToolButton[waveformTool="true"]:hover {
                background: #f4f8fb;
                border-color: #c2d0dc;
            }
            QToolButton[waveformTool="true"]:checked {
                background: #e4edf4;
                border-color: #92a9bd;
                color: #203344;
            }
            QToolButton[waveformTool="true"]:pressed {
                background: #dde7ef;
            }
            QToolButton[waveformTool="true"]:disabled {
                background: #f9fafb;
                border-color: #e4e8ed;
                color: #a7b1ba;
            }
            QFrame[waveformRow="true"] {
                background: transparent;
                border: none;
            }
            QFrame[waveformSlot="true"] {
                background: transparent;
                border: none;
            }
            QFrame[toolbarDivider="true"] {
                background: transparent;
                border: none;
            }
            QLabel[waveformBadge="true"], QToolButton[waveformBadge="true"] {
                color: #425665;
                background: #f5f8fb;
                border: 1px solid #dbe3ea;
                min-height: 24px;
                border-radius: 7px;
                padding-left: 4px;
                padding-right: 2px;
                font-size: 9px;
                font-weight: 700;
            }
            QToolButton[waveformBadge="true"] {
                min-width: 36px;
            }
            QToolButton[waveformBadge="true"]:hover {
                background: #edf4f9;
                border-color: #cad8e3;
                color: #30495b;
            }
            QToolButton[waveformBadge="true"]:pressed {
                background: #e6eef5;
            }
            QToolButton[waveformBadge="true"]:disabled {
                color: #8fa0ad;
                background: #f8fafc;
                border-color: #e2e8ee;
            }
            QLabel[waveformKeyLabel="true"] {
                color: #506474;
                background: #edf2f6;
                border: 1px solid #d8e0e7;
                border-radius: 5px;
                padding: 2px 5px;
                font-size: 8px;
                font-weight: 700;
            }
            QToolButton[waveformPreset="true"] {
                min-width: 36px;
                min-height: 26px;
                padding: 0 5px;
                border-radius: 7px;
                border: 1px solid #dbe3ea;
                background: #ffffff;
                color: #4c6172;
                font-size: 8px;
                font-weight: 600;
            }
            QToolButton[waveformPreset="true"]:hover {
                background: #f5f8fb;
                border-color: #c7d4df;
                color: #2f4758;
            }
            QToolButton[waveformPreset="true"]:checked {
                background: #e9f0f6;
                border-color: #95acbf;
                color: #203547;
            }
            QAbstractSpinBox[waveformField="true"] {
                min-height: 24px;
                padding: 0 4px;
                border-radius: 7px;
                border: 1px solid #d7e0e7;
                background: #fbfcfd;
                color: #2a3f50;
            }
            QAbstractSpinBox[waveformField="true"]:focus {
                border-color: #8ea5b8;
                background: #ffffff;
            }
            QAbstractSpinBox[waveformField="true"][waveformStepper="false"]::up-button,
            QAbstractSpinBox[waveformField="true"][waveformStepper="false"]::down-button {
                width: 0px;
                border: none;
            }
            QAbstractSpinBox[waveformField="true"][waveformStepper="true"]::up-button,
            QAbstractSpinBox[waveformField="true"][waveformStepper="true"]::down-button {
                width: 16px;
                border: none;
                background: transparent;
            }
            QToolButton::menu-indicator {
                width: 10px;
            }
        """ + waveform_stepper_stylesheet + checkbox_stylesheet + """
            QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QComboBox {
                border: 1px solid #cfd6dd;
                border-radius: 5px;
                background: #ffffff;
                padding: 1px 3px;
                font-size: 9px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus, QComboBox:focus {
                border: 1px solid #7f97aa;
                background: #ffffff;
            }
            QLineEdit[readOnlyField="true"], QTextEdit[readOnlyField="true"] {
                background: #f5f7f9;
                color: #556472;
                border-color: #d7dde4;
            }
            QTextEdit[consolePanel="true"] {
                background: #f7f9fb;
                color: #334455;
                border: 1px solid #d4dbe2;
                border-radius: 8px;
                font-family: Menlo, Monaco, monospace;
                font-size: 9px;
            }
            QTableWidget {
                background: #ffffff;
                border: 1px solid #d8dde3;
                border-radius: 7px;
                gridline-color: #edf0f2;
                font-size: 9px;
            }
            QTableWidget[summaryTable="true"] {
                background: #ffffff;
                border: 1px solid #d8dde3;
                border-radius: 8px;
                gridline-color: #edf0f2;
            }
            QTableWidget[summaryTable="true"]::item:selected {
                background: #e7eef5;
                color: #243746;
            }
            QTableWidget[summaryTable="true"] QHeaderView::section {
                background: #f5f7f9;
                color: #556472;
                border: none;
                border-bottom: 1px solid #e5e9ee;
                padding: 4px 5px;
                font-weight: 600;
                font-size: 8px;
            }
            QHeaderView::section {
                background: #f5f7f9;
                color: #556472;
                border: none;
                border-bottom: 1px solid #e5e9ee;
                padding: 2px 3px;
                font-weight: 600;
                font-size: 8px;
            }
            QTableWidget::item:selected {
                background: #e7eef5;
                color: #243746;
            }
            QTabWidget::pane {
                border: 1px solid #d8dde3;
                border-radius: 7px;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #eef1f4;
                color: #5b6874;
                padding: 3px 6px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 2px;
                font-size: 9px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #243746;
                font-weight: 600;
            }
            QTabWidget[inspectorTabs="true"]::pane {
                border: none;
                top: -1px;
                background: transparent;
            }
            QTabWidget[inspectorTabs="true"] QTabBar::tab {
                background: #eef2f5;
                color: #5b6d7a;
                padding: 5px 9px;
                margin-right: 3px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                min-height: 20px;
            }
            QTabWidget[inspectorTabs="true"] QTabBar::tab:selected {
                background: #ffffff;
                color: #23384a;
                font-weight: 700;
            }
            QTabWidget[inspectorTabs="true"] QTabBar::tab:hover {
                color: #31495c;
            }
            QSplitter#loadedWorkspaceSplitter::handle {
                background: #e6edf2;
            }
            QSplitter#loadedWorkspaceSplitter::handle:vertical {
                height: 8px;
            }
            QFrame#WaveformHeader {
                background: transparent;
                border: none;
                border-bottom: 1px solid #d8dde3;
            }
            QFrame#workflowHeader, QFrame#workflowSteps {
                background: #f7f8fa;
                border: 1px solid #dde2e8;
                border-radius: 10px;
                color: #253746;
            }
            QFrame#InspectorContent {
                background: #f4f6f8;
                border-left: 1px solid #dfe4e8;
            }
            QFrame#InspectorSection {
                background: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
            }
            QFrame[inspectorHeader="true"] {
                background: transparent;
                border: none;
                border-bottom: 1px solid #ddd6cc;
            }
            QFrame[inspectorHero="true"] {
                background: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
            }
            QFrame[inspectorCard="true"] {
                background: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
            }
            QFrame[inspectorChip="true"] {
                background: #f8f9fa;
                border: 1px solid #e3e6ea;
                border-radius: 4px;
            }
            QFrame[inspectorAccent="true"] {
                background: #4d90fe;
                border: none;
                border-radius: 1px;
            }
            QFrame[inspectorRule="true"] {
                background: #e9edf2;
                border: none;
                min-height: 1px;
                max-height: 1px;
            }
            QFrame[metricCard="true"] {
                background: #ffffff;
                border: 1px solid #dde2e8;
                border-radius: 10px;
            }
            QFrame[metricCard="true"][waveformToolbarShell="true"] {
                background: #fdfefe;
                border: 1px solid #d8e1e8;
                border-radius: 14px;
            }
            QFrame[softCard="true"] {
                background: #fafbfc;
                border: 1px solid #e5e9ee;
                border-radius: 8px;
            }
            QFrame[legendChip="true"] {
                background: rgba(244, 247, 250, 0.92);
                border: 1px solid rgba(36, 55, 70, 0.12);
                border-radius: 7px;
            }
            QFrame[legendSwatch="true"] {
                border: 1px solid rgba(36, 55, 70, 0.18);
                border-radius: 3px;
            }
            QFrame[statusDot="true"][activityLevel="info"] {
                background: #49667d;
                border-radius: 5px;
            }
            QFrame[statusDot="true"][activityLevel="warning"] {
                background: #9a6a2b;
                border-radius: 5px;
            }
            QFrame[statusDot="true"][activityLevel="done"] {
                background: #4f7a57;
                border-radius: 5px;
            }
            QFrame[statusDot="true"][activityLevel="error"] {
                background: #a5534a;
                border-radius: 5px;
            }
            QFrame#CompactField {
                background: transparent;
                border: none;
            }
            QScrollArea#InspectorScrollArea {
                background: transparent;
                border: none;
            }
            QScrollArea#InspectorScrollArea > QWidget > QWidget {
                background: transparent;
            }
            QLabel[inspectorEyebrow="true"] {
                color: #7a8794;
                font-size: 8px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QLabel[inspectorDockTitle="true"] {
                color: #333333;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel[inspectorBodyText="true"] {
                color: #6b7280;
                font-size: 9px;
            }
            QLabel[inspectorHeroTitle="true"] {
                color: #333333;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel[inspectorHeroMeta="true"] {
                color: #6b7280;
                font-size: 8px;
            }
            QLabel[inspectorCardHeading="true"] {
                color: #333333;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel[inspectorSectionTitle="true"] {
                color: #333333;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel[inspectorSectionSubtitle="true"] {
                color: #7b8792;
                font-size: 8px;
            }
            QLabel[inspectorChipLabel="true"] {
                color: #7b8792;
                font-size: 8px;
                font-weight: 700;
            }
            QLabel[inspectorChipValue="true"] {
                color: #333333;
                font-size: 9px;
                font-weight: 700;
            }
            QLabel[fieldLabel="true"] {
                color: #4b5563;
                font-size: 9px;
                font-weight: 600;
            }
            QLabel[fieldUnit="true"] {
                color: #7b8792;
                font-size: 8px;
            }
            QLabel[toolbarLabel="true"] {
                color: #6f7b84;
                font-size: 8px;
                font-weight: 700;
            }
            QLabel[dataBadge="true"] {
                background: #ffffff;
                border: 1px solid #d8dde3;
                border-radius: 8px;
                padding: 4px 8px;
                color: #243746;
                font-size: 9px;
                font-weight: 600;
            }
            QLabel[pageEyebrow="true"] {
                color: #73808b;
                font-size: 11px;
                font-weight: 700;
            }
            QLabel[pageTitle="true"] {
                color: #253746;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel[heroTitle="true"] {
                color: #253746;
                font-size: 18px;
                font-weight: 700;
            }
            QLabel[pageSubtitle="true"] {
                color: #6f7a84;
                font-size: 12px;
            }
            QLabel[heroSubtitle="true"] {
                color: #6a7681;
                font-size: 11px;
            }
            QLabel[sectionTitle="true"] {
                color: #1f3448;
                font-size: 13px;
                font-weight: 700;
            }
            QLabel[panelTitle="true"] {
                color: #1f3448;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel[sectionSubtitle="true"], QLabel[helperText="true"] {
                color: #687888;
                font-size: 10px;
            }
            QLabel[metricLabel="true"] {
                color: #6a7784;
                font-size: 10px;
                font-weight: 600;
            }
            QLabel[metricValue="true"] {
                color: #1f3448;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel[stepBadge="true"] {
                background: #e7edf2;
                color: #435565;
                border-radius: 11px;
                font-size: 10px;
                font-weight: 700;
                padding: 0 6px;
            }
            QLabel[stepTitle="true"] {
                color: #34485a;
                font-size: 11px;
                font-weight: 700;
            }
            QLabel[stepDescription="true"] {
                color: #7a8792;
                font-size: 9px;
            }
            QLabel[legendLabel="true"] {
                color: #304657;
                font-size: 8px;
                font-weight: 600;
            }
            QLabel[waveformHeaderTitle="true"] {
                color: #243746;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel[waveformHeaderMeta="true"] {
                color: #627180;
                font-size: 7px;
            }
            QLabel[waveformHeaderMetaStrong="true"] {
                color: #466074;
                font-size: 9px;
                font-weight: 600;
            }
            QLabel[workflowStatus="true"] {
                color: #4f6272;
                font-size: 11px;
                font-weight: 600;
                padding-top: 1px;
            }
            QLabel[chip="true"] {
                background: #eef2f6;
                color: #5b6b78;
                border: 1px solid #d8e0e7;
                border-radius: 11px;
                padding: 2px 9px;
                font-size: 9px;
                font-weight: 600;
            }
            QLabel[chip="true"][active="true"] {
                background: #dbe7f0;
                color: #274256;
                border-color: #bcd0dd;
            }
            QFrame#WaveformPlaceholder {
                background: rgba(255, 255, 255, 244);
                border: 1px solid #d8dde3;
                border-radius: 10px;
            }
            QLabel[secondaryText="true"] {
                color: #64717d;
            }
            QLabel[statusValue="true"] {
                font-size: 11px;
                font-weight: 700;
                color: #243746;
            }
            #InspectorSinglePage {
                background: transparent;
            }
            #InspectorSinglePage QWidget[inspectorSectionBody="true"] {
                background: transparent;
            }
            #InspectorSinglePage QLabel {
                font-size: 9px;
                color: #49515a;
            }
            #InspectorSinglePage QGroupBox {
                font-size: 10px;
                font-weight: 700;
                color: #333333;
                margin-top: 0px;
                border: none;
                background: transparent;
                padding-top: 0px;
            }
            #InspectorSinglePage QGroupBox::title {
                subcontrol-origin: margin;
                left: 0px;
                padding: 0;
                color: #333333;
            }
            #InspectorSinglePage QPushButton,
            #InspectorSinglePage QToolButton,
            #InspectorSinglePage QLineEdit,
            #InspectorSinglePage QSpinBox,
            #InspectorSinglePage QDoubleSpinBox,
            #InspectorSinglePage QComboBox {
                font-size: 9px;
                min-height: """ + f"{self.ui_density.compact_input_height}" + """px;
                background: #f8f9fa;
                border: 1px solid #d9dee3;
                border-radius: 4px;
                padding: 1px 5px;
                color: #2f3740;
            }
            #InspectorSinglePage QLineEdit:focus,
            #InspectorSinglePage QSpinBox:focus,
            #InspectorSinglePage QDoubleSpinBox:focus,
            #InspectorSinglePage QComboBox:focus {
                border: 1px solid #4d90fe;
                background: #ffffff;
            }
            #InspectorSinglePage QLineEdit[readOnly='true'] {
                background: #f2f4f6;
                color: #5e6975;
            }
            #InspectorSinglePage QComboBox {
                padding-right: 23px;
            }
            #InspectorSinglePage QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 18px;
                border: none;
                border-left: 1px solid #dde3e9;
                background: #eef2f6;
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
            }
            #InspectorSinglePage QAbstractSpinBox::up-button,
            #InspectorSinglePage QAbstractSpinBox::down-button {
                width: 16px;
                border: none;
                background: transparent;
            }
            #InspectorSinglePage QCheckBox {
                font-size: 9px;
            }
        """ + inspector_checkbox_stylesheet + """
            #InspectorSinglePage QTabWidget::pane {
                border: none;
                background: transparent;
            }
            #InspectorSinglePage QTabBar::tab {
                padding: 0px;
                margin: 0px;
                min-height: 0px;
                min-width: 0px;
                max-width: 0px;
                border: none;
                background: transparent;
                color: transparent;
            }
            #InspectorSinglePage QTableWidget {
                background: #ffffff;
                border: 1px solid #e1e5ea;
                border-radius: 4px;
                gridline-color: #eef1f4;
                font-size: 10px;
            }
            #InspectorSinglePage QHeaderView::section {
                background: #f8f9fa;
                color: #6b7280;
                border: none;
                border-bottom: 1px solid #e5e7eb;
                padding: 3px 4px;
                font-weight: 700;
                font-size: 8px;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 8px;
                margin: 2px 0 2px 0;
            }
            QScrollBar::handle:vertical {
                background: #c8d1d7;
                border-radius: 4px;
                min-height: 22px;
            }
            QScrollBar::handle:vertical:hover {
                background: #afbcc5;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """ + popup_stylesheet + f"""
            QPushButton {{
                min-height: {button_min_height}px;
                padding-top: 0px;
                padding-bottom: 0px;
            }}
            QToolButton {{
                min-height: {button_min_height}px;
                padding-top: 0px;
                padding-bottom: 0px;
            }}
            QToolButton[inspectorMenu="true"] {{
                min-height: {tool_min_height}px;
            }}
            QToolButton[waveformTool="true"] {{
                min-height: {tool_min_height}px;
            }}
            QToolButton[waveformPreset="true"] {{
                min-height: {tool_min_height}px;
            }}
            QLabel[legendLabel="true"] {{
                font-size: {legend_label_font}px;
            }}
            QLabel[waveformHeaderTitle="true"] {{
                font-size: {waveform_header_title_font}px;
            }}
            QLabel[waveformHeaderMeta="true"] {{
                font-size: {waveform_header_meta_font}px;
            }}
            QLabel[waveformHeaderMetaStrong="true"] {{
                font-size: {waveform_header_meta_strong_font}px;
            }}
        """)

    def _configure_hub_model_banner(self):
        label = getattr(self.window, "image_label", None)
        if label is None:
            return
        label.setText("")
        label.setAlignment(Qt.AlignCenter)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        label.setMinimumHeight(108)
        label.setMaximumHeight(132)
        label.setMargin(0)
        label.setScaledContents(False)
        label.setToolTip("Hosted model cards for the artifact, spk-HFO, and eHFO classifiers.")
        if not bool(label.property("hubBannerFilterInstalled")):
            label.installEventFilter(self)
            label.setProperty("hubBannerFilterInstalled", True)
        self._refresh_hub_model_banner()

    def _load_hub_mascot_pixmap(self):
        cached = getattr(self, "_hub_mascot_pixmap", None)
        if cached is not None:
            return cached
        source = QPixmap(os.path.join(ROOT_DIR, "src/ui/images/huggingface_logo.png"))
        if source.isNull():
            self._hub_mascot_pixmap = QPixmap()
            return self._hub_mascot_pixmap
        crop_width = min(source.height(), source.width())
        self._hub_mascot_pixmap = source.copy(0, 0, crop_width, source.height())
        return self._hub_mascot_pixmap

    def _build_hub_model_banner(self, width, height):
        canvas = QtGui.QPixmap(width, height)
        canvas.fill(Qt.transparent)
        painter = QtGui.QPainter(canvas)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        compact = width < 480

        outer_rect = QtCore.QRectF(0.5, 0.5, width - 1, height - 1)
        card_path = QtGui.QPainterPath()
        card_path.addRoundedRect(outer_rect, 18, 18)
        card_gradient = QtGui.QLinearGradient(0, 0, width, height)
        card_gradient.setColorAt(0.0, QtGui.QColor("#fff7ea"))
        card_gradient.setColorAt(0.42, QtGui.QColor("#fffdfb"))
        card_gradient.setColorAt(1.0, QtGui.QColor("#f3f8fc"))
        painter.fillPath(card_path, card_gradient)
        painter.setPen(QtGui.QPen(QtGui.QColor("#e1e7ec"), 1))
        painter.drawPath(card_path)

        accent_rect = QtCore.QRectF(16, 14, 86 if compact else 118, height - 28)
        accent_gradient = QtGui.QLinearGradient(accent_rect.topLeft(), accent_rect.bottomRight())
        accent_gradient.setColorAt(0.0, QtGui.QColor("#ffd56c"))
        accent_gradient.setColorAt(1.0, QtGui.QColor("#ffaf3f"))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(accent_gradient)
        painter.drawRoundedRect(accent_rect, 24, 24)

        painter.setBrush(QtGui.QColor(255, 255, 255, 55))
        painter.drawRoundedRect(
            QtCore.QRectF(22, 20, 38 if compact else 54, 18 if compact else 22),
            11,
            11,
        )

        badge_size = min(accent_rect.width() - (26 if compact else 32), accent_rect.height() - 20)
        badge_rect = QtCore.QRectF(
            accent_rect.center().x() - (badge_size / 2),
            accent_rect.center().y() - (badge_size / 2),
            badge_size,
            badge_size,
        )
        painter.setBrush(QtGui.QColor(255, 252, 244, 210))
        painter.drawEllipse(badge_rect)

        mascot = self._load_hub_mascot_pixmap()
        if not mascot.isNull():
            target_width = int(badge_rect.width() * 0.82)
            target_height = int(badge_rect.height() * 0.82)
            scaled_mascot = mascot.scaled(
                target_width,
                target_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            mascot_x = int(badge_rect.center().x() - (scaled_mascot.width() / 2))
            mascot_y = int(badge_rect.center().y() - (scaled_mascot.height() / 2))
            painter.drawPixmap(mascot_x, mascot_y, scaled_mascot)
        else:
            fallback_font = QtGui.QFont(self.window.font())
            fallback_font.setBold(True)
            fallback_font.setPixelSize(26)
            painter.setFont(fallback_font)
            painter.setPen(QtGui.QColor("#6b4700"))
            painter.drawText(badge_rect, Qt.AlignCenter, "HF")

        text_left = int(accent_rect.right()) + (14 if compact else 20)
        text_width = max(120, width - text_left - 18)

        title_font = QtGui.QFont(self.window.font())
        title_font.setBold(True)
        title_font.setPixelSize(18 if width < 300 else 20 if compact else 26 if width < 620 else 30)
        subtitle_font = QtGui.QFont(self.window.font())
        subtitle_font.setPixelSize(11 if compact else 13 if width < 620 else 14)

        title_metrics = QtGui.QFontMetrics(title_font)
        subtitle_metrics = QtGui.QFontMetrics(subtitle_font)
        title_source = "HF Hub" if width < 300 else "Hugging Face" if compact else "Hugging Face Hub"
        subtitle_source = (
            "Artifact, spk-HFO, eHFO"
            if compact
            else "Hosted presets for artifact, spk-HFO, and eHFO classifiers."
        )
        title_text = title_metrics.elidedText(title_source, Qt.ElideRight, text_width)
        subtitle_text = subtitle_metrics.elidedText(
            subtitle_source,
            Qt.ElideRight,
            text_width,
        )

        painter.setFont(title_font)
        painter.setPen(QtGui.QColor("#1f2f3d"))
        painter.drawText(
            QtCore.QRectF(text_left, 26 if compact else 24, text_width, 30 if compact else 34),
            Qt.AlignLeft | Qt.AlignVCenter,
            title_text,
        )
        painter.setFont(subtitle_font)
        painter.setPen(QtGui.QColor("#5c6f7f"))
        painter.drawText(
            QtCore.QRectF(text_left, 54 if compact else 58, text_width, 16 if compact else 20),
            Qt.AlignLeft | Qt.AlignVCenter,
            subtitle_text,
        )

        pill_text = "Model cards"
        pill_font = QtGui.QFont(self.window.font())
        pill_font.setBold(True)
        pill_font.setPixelSize(12)
        pill_metrics = QtGui.QFontMetrics(pill_font)
        pill_width = pill_metrics.horizontalAdvance(pill_text) + 20
        if width >= 560:
            pill_rect = QtCore.QRectF(width - pill_width - 18, 18, pill_width, 24)
            painter.setBrush(QtGui.QColor("#fff4d6"))
            painter.setPen(QtGui.QPen(QtGui.QColor("#f0c96f"), 1))
            painter.drawRoundedRect(pill_rect, 12, 12)
            painter.setFont(pill_font)
            painter.setPen(QtGui.QColor("#8e6411"))
            painter.drawText(pill_rect, Qt.AlignCenter, pill_text)

        painter.end()
        return canvas

    def _refresh_hub_model_banner(self):
        label = getattr(self.window, "image_label", None)
        if label is None:
            return
        available_width = label.width()
        if available_width <= 0:
            group_box = getattr(self.window, "groupBox_2", None)
            if group_box is not None:
                available_width = max(0, group_box.width() - 28)
        available_width = max(240, min(760, available_width))
        label.setPixmap(self._build_hub_model_banner(available_width, 108))

    def _configure_native_toolbars(self):
        toolbar = getattr(self.window, "toolBar", None)
        if toolbar is None:
            return

        style = self.window.style()
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setIconSize(QtCore.QSize(16, 16))
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        action_specs = [
            ("actionOpen_EDF_toolbar", "Open", "Open a recording", style.standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)),
            ("actionLoad_Detection_toolbar", "Session", "Load a saved PyBrain session", style.standardIcon(QtWidgets.QStyle.SP_DirOpenIcon)),
            ("actionQuick_Detection_toolbar", "Quick", "Open quick detection", style.standardIcon(QtWidgets.QStyle.SP_MediaPlay)),
        ]
        for action_name, text, tooltip, icon in action_specs:
            action = getattr(self.window, action_name, None)
            if action is None:
                continue
            action.setText(text)
            action.setToolTip(tooltip)
            action.setStatusTip(tooltip)
            action.setIcon(icon)

        if not hasattr(self.window, "actionWaveform_Shortcuts_toolbar"):
            shortcuts_action = QtWidgets.QAction(
                style.standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView),
                "Shortcuts",
                self.window,
            )
            shortcuts_action.setObjectName("actionWaveform_Shortcuts_toolbar")
            self.window.actionWaveform_Shortcuts_toolbar = shortcuts_action
        self.window.actionWaveform_Shortcuts_toolbar.setToolTip("Adjust waveform shortcut settings")
        self.window.actionWaveform_Shortcuts_toolbar.setStatusTip("Adjust waveform shortcut settings")
        if self.window.actionWaveform_Shortcuts_toolbar not in toolbar.actions():
            toolbar.addAction(self.window.actionWaveform_Shortcuts_toolbar)

    def _install_report_actions(self):
        if not hasattr(self.window, "save_report_button"):
            self.window.save_report_button = QPushButton("Export Report")
            self.window.save_report_button.setToolTip("Export a shareable HTML report package for the accepted or active run")
            layout = getattr(self.window, "gridLayout_17", None)
            if layout is not None:
                layout.addWidget(self.window.save_report_button, 1, 2)

    def _build_native_window_shell(self):
        waveform_widget = self.window.waveformWiget
        inspector_tabs = self.window.tabWidget
        recording_info = self.window.main_edf_info
        terminal = self.window.STDTextEdit

        waveform_widget.setParent(None)
        inspector_tabs.setParent(None)
        recording_info.setParent(None)
        terminal.setParent(None)

        old_central = self.window.centralWidget()
        root_layout = getattr(old_central, "layout", lambda: None)()
        if root_layout is not None:
            while root_layout.count():
                item = root_layout.takeAt(0)
                child = item.widget()
                if child is not None:
                    child.setParent(None)

        central = QWidget(self.window)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(10, 10, 10, 10)
        central_layout.setSpacing(8)
        self.window.setCentralWidget(central)

        header = self._build_waveform_header()
        central_layout.addWidget(header)
        central_layout.addWidget(waveform_widget, 1)

        self.window.waveform_header_frame = header
        self.window.waveformWiget = waveform_widget
        self.window.tabWidget = inspector_tabs
        self.window.main_edf_info = recording_info
        self.window.STDTextEdit = terminal

        self._relocate_biomarker_legend_to_header()
        self._detach_activity_strip()
        self._create_waveform_placeholder()
        self._build_run_statistics_dialog()
        self._build_inspector_dock()
        self._build_status_strip()

    def _apply_default_dock_sizes(self):
        if hasattr(self.window, "inspector_dock"):
            self.window.resizeDocks([self.window.inspector_dock], [self.ui_density.inspector_dock_width], Qt.Horizontal)

    def _build_waveform_header(self):
        header = QFrame(self.window)
        header.setObjectName("WaveformHeader")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(21, 4, 17, 6)
        layout.setSpacing(4)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)

        title = QLabel("EEG/iEEG Review")
        title.setProperty("waveformHeaderTitle", True)
        self.window.waveform_header_title = title
        top_row.addWidget(title, 0, Qt.AlignLeft | Qt.AlignVCenter)
        top_row.addSpacing(12)

        self.window.legend_host_frame = QFrame(header)
        self.window.legend_host_frame.setObjectName("LegendHostFrame")
        self.window.legend_host_frame.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.window.legend_host_frame.setMinimumWidth(0)
        top_row.addWidget(self.window.legend_host_frame, 0, Qt.AlignLeft | Qt.AlignVCenter)
        top_row.addStretch(1)
        layout.addLayout(top_row)

        subtitle = QLabel("")
        subtitle.setProperty("waveformHeaderMeta", True)
        self.window.workflow_status_label = subtitle
        subtitle.setVisible(False)
        self.window.recording_header_label = QLabel("")
        self.window.recording_header_label.setProperty("waveformHeaderMetaStrong", True)
        self.window.recording_header_label.setVisible(False)
        layout.addWidget(subtitle)
        layout.addWidget(self.window.recording_header_label)

        mode_frame = QFrame(header)
        mode_frame.setVisible(False)
        mode_layout = QHBoxLayout(mode_frame)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(6)

        self.window.waveform_source_mode_badge = QLabel("Source --", mode_frame)
        self.window.waveform_source_mode_badge.setProperty("waveformBadge", True)
        mode_layout.addWidget(self.window.waveform_source_mode_badge, 0, Qt.AlignLeft | Qt.AlignVCenter)

        self.window.waveform_scope_mode_badge = QLabel("Scope --", mode_frame)
        self.window.waveform_scope_mode_badge.setProperty("waveformBadge", True)
        mode_layout.addWidget(self.window.waveform_scope_mode_badge, 0, Qt.AlignLeft | Qt.AlignVCenter)

        self.window.waveform_tool_mode_badge = QLabel("Tool --", mode_frame)
        self.window.waveform_tool_mode_badge.setProperty("waveformBadge", True)
        mode_layout.addWidget(self.window.waveform_tool_mode_badge, 0, Qt.AlignLeft | Qt.AlignVCenter)

        mode_layout.addStretch(1)

        self.window.waveform_reset_view_button = QToolButton(mode_frame)
        self.window.waveform_reset_view_button.setText("Reset View")
        self.window.waveform_reset_view_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.window.waveform_reset_view_button.setProperty("waveformBadge", True)
        self.window.waveform_reset_view_button.setCursor(Qt.PointingHandCursor)
        self.window.waveform_reset_view_button.setEnabled(False)
        self.window.waveform_reset_view_button.setToolTip(
            "Return to the referential source channels and clear sticky waveform tools."
        )
        mode_layout.addWidget(self.window.waveform_reset_view_button, 0, Qt.AlignRight | Qt.AlignVCenter)

        self.window.waveform_mode_frame = mode_frame
        layout.addWidget(mode_frame)
        return header

    def _relocate_biomarker_legend_to_header(self):
        legend_frame = self.window.frame_biomarker_type
        legend_frame.setParent(self.window.legend_host_frame)
        legend_frame.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        legend_frame.setMinimumWidth(0)
        host_layout = QHBoxLayout(self.window.legend_host_frame)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(4)
        host_layout.addWidget(legend_frame, 0, Qt.AlignLeft | Qt.AlignVCenter)
        host_layout.addStretch(1)
        legend_frame.setFrameShape(QFrame.NoFrame)

    def _detach_activity_strip(self):
        waveform_layout = self.window.waveformWiget.layout()
        if waveform_layout.count() > 0:
            last_index = waveform_layout.count() - 1
            trailing_item = waveform_layout.itemAt(last_index)
            if trailing_item is not None and trailing_item.layout() is getattr(self.window, "gridLayout_6", None):
                waveform_layout.takeAt(last_index)

        waveform_layout.setSpacing(2)
        waveform_layout.setStretch(0, 0)
        waveform_layout.setStretch(1, 0)
        waveform_layout.setStretch(2, 1)
        self.window.widget.setMinimumHeight(280)

    def _refine_waveform_control_strip(self):
        density = self.ui_density
        top_controls = getattr(self.window, "horizontalLayout_11", None)
        if top_controls is not None:
            top_controls.setContentsMargins(7, 0, 4, 0)
            top_controls.setSpacing(3)

        secondary_controls = getattr(self.window, "horizontalLayout_14", None)
        if secondary_controls is not None:
            secondary_controls.setContentsMargins(7, 0, 4, 0)
            secondary_controls.setSpacing(2)

        jobs_layout = getattr(self.window, "horizontalLayout_13", None)
        if jobs_layout is not None:
            jobs_layout.setContentsMargins(4, 0, 0, 0)
            jobs_layout.setSpacing(4)

        control_strip_height = max(96, density.waveform_control_strip_height + 4)
        self.window.widget_2.setMinimumHeight(control_strip_height)
        self.window.widget_2.setMaximumHeight(control_strip_height)
        self.window.n_channel_input.setMaximumWidth(70)
        self.window.display_time_window_input.setMaximumWidth(128)
        self.window.Time_Increment_Input.setMaximumWidth(80)
        self.window.n_jobs_spinbox.setMaximumWidth(54)
        self.window.label_10.setText("Visible")
        self.window.label_10.setToolTip("How many channels are shown at once in the waveform")
        self.window.label_10.setProperty("waveformKeyLabel", True)
        self.window.label_10.setAlignment(Qt.AlignCenter)
        self.window.label_9.setText("Window")
        self.window.label_9.setToolTip("Length of the waveform view")
        self.window.label_9.setProperty("waveformKeyLabel", True)
        self.window.label_9.setAlignment(Qt.AlignCenter)
        self.window.display_time_window_input.setSuffix(" s")
        self.window.display_time_window_input.setToolTip("Length of the waveform window in seconds")
        self.window.display_time_window_input.setKeyboardTracking(False)
        self.window.label_8.hide()
        self.window.label_14.setText("Advance")
        self.window.label_14.setToolTip("How far the waveform view moves when stepping through time")
        self.window.label_14.setProperty("waveformKeyLabel", True)
        self.window.label_14.setAlignment(Qt.AlignCenter)
        self.window.Time_Increment_Input.setSuffix("%")
        self.window.Time_Increment_Input.setToolTip("Percent of the current window to move when stepping or scrolling")
        self.window.label_22.hide()

        for widget, prefix in (
            (self.window.display_time_window_input, "Win "),
            (self.window.Time_Increment_Input, "Step "),
            (self.window.n_channel_input, "Vis "),
        ):
            widget.setPrefix(prefix)
            widget.setAlignment(Qt.AlignCenter)
            if widget is self.window.display_time_window_input:
                widget.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
                widget.setProperty("waveformStepper", True)
            else:
                widget.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
                widget.setProperty("waveformStepper", False)
            widget.setProperty("waveformField", True)

        for label in (self.window.label_10, self.window.label_9, self.window.label_14):
            label.hide()

        for label in (
            self.window.label_10,
            self.window.label_9,
            self.window.label_8,
            self.window.label_14,
            self.window.label_22,
        ):
            if label in (self.window.label_10, self.window.label_9, self.window.label_14):
                continue
            label.setProperty("toolbarLabel", True)

        for widget in (
            self.window.bipolar_button,
            self.window.Choose_Channels_Button,
            self.window.waveform_plot_button,
        ):
            widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        self._build_waveform_toolbar()

    def _move_runtime_controls_to_prepare_panel(self):
        filter_layout = getattr(self.window.filters_groupbox, "layout", lambda: None)()
        if filter_layout is None:
            return
        if hasattr(self.window, "line"):
            self.window.line.hide()

        runtime_frame = QFrame(self.window.filters_groupbox)
        runtime_layout = QHBoxLayout(runtime_frame)
        runtime_layout.setContentsMargins(0, 1, 0, 0)
        runtime_layout.setSpacing(6)

        self.window.label_21.setParent(runtime_frame)
        self.window.label_21.setText("Workers")
        self.window.label_21.setProperty("toolbarLabel", True)
        self.window.n_jobs_spinbox.setParent(runtime_frame)
        self.window.n_jobs_ok_button.setParent(runtime_frame)
        self.window.n_jobs_spinbox.setMaximumWidth(56)
        self.window.n_jobs_ok_button.setText("Set")
        self.window.n_jobs_ok_button.setMaximumWidth(48)

        runtime_layout.addWidget(self.window.label_21)
        runtime_layout.addWidget(self.window.n_jobs_spinbox)
        runtime_layout.addWidget(self.window.n_jobs_ok_button)
        runtime_layout.addStretch(1)

        filter_layout.addWidget(runtime_frame, 1, 0)
        self.window.prepare_runtime_frame = runtime_frame

    def _enhance_analysis_setup_tabs(self):
        self._build_detector_mode_header()
        self._build_classifier_mode_header()

    def _streamline_prepare_tab(self):
        if hasattr(self.window, "stacked_widget_detection_param"):
            self.window.stacked_widget_detection_param.setVisible(False)
        if hasattr(self.window, "classifier_groupbox_4"):
            self.window.classifier_groupbox_4.setVisible(False)
        if hasattr(self.window, "statistics_box"):
            self.window.statistics_box.setTitle("Outputs")
            self.window.statistics_box.setMaximumHeight(70)
            self.window.statistics_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            stats_layout = getattr(self.window.statistics_box, "layout", lambda: None)()
            if isinstance(stats_layout, QGridLayout):
                stats_layout.setRowStretch(0, 0)
                stats_layout.setRowStretch(1, 0)
                stats_layout.setVerticalSpacing(4)
                stats_layout.setContentsMargins(6, 8, 6, 6)
        if hasattr(self.window, "statistics_label"):
            self.window.statistics_label.setVisible(False)
        if hasattr(self.window, "annotation_button"):
            self.window.annotation_button.setText("Open Review")
            self.window.annotation_button.setToolTip("Open the event-by-event review window for the active run")
        if hasattr(self.window, "save_npz_button"):
            self.window.save_npz_button.setText("Save Session")
            self.window.save_npz_button.setToolTip("Save the current PyBrain session")
        if hasattr(self.window, "save_csv_button"):
            self.window.save_csv_button.setText("Export Workbook")
            self.window.save_csv_button.setToolTip("Export the current accepted result workbook")
        if hasattr(self.window, "overview_filter_button"):
            self.window.overview_filter_button.setText("Filter")
            self.window.overview_filter_button.setMinimumWidth(72)
        for button in (
            getattr(self.window, "annotation_button", None),
            getattr(self.window, "save_npz_button", None),
            getattr(self.window, "save_csv_button", None),
        ):
            if button is not None:
                button.setMinimumHeight(self.ui_density.compact_button_height)
                button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        overview_layout = getattr(self.window.overview_tab, "layout", lambda: None)()
        if overview_layout is not None and not hasattr(self.window, "prepare_tab_compact_container"):
            compact_container = QWidget(self.window.overview_tab)
            compact_layout = QVBoxLayout(compact_container)
            compact_layout.setContentsMargins(0, 0, 0, 0)
            compact_layout.setSpacing(3)
            signal_stack = getattr(self.window, "stackedWidget_2", None)
            outputs_stack = getattr(self.window, "stackedWidget_4", None)
            if signal_stack is not None:
                signal_stack.setParent(compact_container)
                signal_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                compact_layout.addWidget(signal_stack)
            if outputs_stack is not None:
                outputs_stack.setParent(self.window.overview_tab)
                outputs_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                self.window.prepare_outputs_container = outputs_stack
            compact_layout.setSizeConstraint(QLayout.SetMinimumSize)

            while overview_layout.count():
                item = overview_layout.takeAt(0)
                child = item.widget()
                if child is not None and child is not getattr(self.window, "prepare_outputs_container", None):
                    child.setParent(None)
            overview_layout.addWidget(compact_container, 0, 0)
            overview_layout.setRowStretch(0, 0)
            overview_layout.setRowStretch(1, 0)
            self.window.prepare_tab_compact_container = compact_container

        for stacked_name in ("stackedWidget_2", "stackedWidget_4"):
            stacked = getattr(self.window, stacked_name, None)
            if stacked is not None:
                stacked.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        filter_stack = getattr(self.window, "stackedWidget_2", None)
        if filter_stack is not None:
            filter_stack.setCurrentIndex(0)
            filter_stack.setMaximumHeight(168)
        summary_stack = getattr(self.window, "stackedWidget_4", None)
        if summary_stack is not None:
            summary_stack.setCurrentIndex(0)
            summary_stack.setMaximumHeight(92)

        if hasattr(self.window, "gridLayout_27"):
            self.window.gridLayout_27.setContentsMargins(0, 0, 0, 0)
            self.window.gridLayout_27.setVerticalSpacing(8)

        self._rebuild_filter_section()

    def _build_detector_mode_header(self):
        detector_layout = getattr(self.window, "gridLayout_13", None)
        if detector_layout is None or hasattr(self.window, "detector_mode_combo"):
            return

        detector_layout.setContentsMargins(0, 0, 0, 0)
        detector_layout.setHorizontalSpacing(0)
        detector_layout.setVerticalSpacing(4)
        detector_tabs = self.window.detector_subtabs
        detector_layout.removeWidget(detector_tabs)

        header = QFrame(self.window.detector_tab)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 4)
        header_layout.setSpacing(4)

        label = QLabel("Detector")
        label.setProperty("fieldLabel", True)
        self.window.detector_mode_combo = QComboBox(header)
        self.window.detector_mode_combo.setMinimumWidth(0)
        self.window.detector_apply_button = QPushButton("Apply")
        self.window.detector_apply_button.setMinimumWidth(52)
        self.window.detector_apply_button.setToolTip("Apply the currently visible detector parameters")
        self.window.detector_apply_button.hide()
        self.window.detector_run_button = QPushButton("Run")
        self.window.detector_run_button.setMinimumWidth(48)
        self.window.detector_run_button.setProperty("inspectorPrimary", True)
        self.window.detector_run_button.setToolTip("Apply detector settings and run a new analysis")
        self.window.detector_mode_hint = QLabel("Choose the detector for the next run.")
        self.window.detector_mode_hint.setProperty("helperText", True)
        self.window.detector_mode_hint.setWordWrap(True)
        self.window.detector_mode_hint.hide()

        detector_selector = self._create_compact_control_row(
            "Detector",
            self.window.detector_mode_combo,
            label_widget=label,
            action_widgets=[self.window.detector_run_button],
            option_id="detector_selector",
            density="essential",
        )
        header_layout.addWidget(detector_selector)

        detector_layout.addWidget(header, 0, 0)
        detector_layout.addWidget(detector_tabs, 1, 0)
        self._apply_section_visibility("DETECTION")

    def _build_classifier_mode_header(self):
        classifier_layout = getattr(self.window, "gridLayout_20", None)
        if classifier_layout is None or hasattr(self.window, "classifier_mode_combo"):
            return

        classifier_layout.setContentsMargins(0, 0, 0, 0)
        classifier_layout.setHorizontalSpacing(0)
        classifier_layout.setVerticalSpacing(4)
        header = QFrame(self.window.classifier_tab)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 4)
        header_layout.setSpacing(4)

        label = QLabel("Preset")
        label.setProperty("fieldLabel", True)
        self.window.classifier_mode_combo = QComboBox(header)
        self.window.classifier_mode_combo.setMinimumContentsLength(10)
        self.window.classifier_mode_combo.setMinimumWidth(104)
        self.window.classifier_apply_button = QPushButton("Apply")
        self.window.classifier_apply_button.setMinimumWidth(64)
        self.window.classifier_apply_button.setToolTip("Apply the custom model sources and classifier settings")
        self.window.classifier_run_button = QPushButton("Classify")
        self.window.classifier_run_button.setMinimumWidth(76)
        self.window.classifier_run_button.setProperty("inspectorPrimary", True)
        self.window.classifier_run_button.setToolTip("Run classification on the active run")
        self.window.classifier_mode_hint = QLabel("Choose how classification should run for the next analysis.")
        self.window.classifier_mode_hint.setProperty("helperText", True)
        self.window.classifier_mode_hint.setWordWrap(True)
        self.window.classifier_mode_hint.hide()
        self.window.default_cpu_button.setVisible(False)
        self.window.default_gpu_button.setVisible(False)
        self.window.classifier_save_button.setVisible(False)
        self.window.classifier_apply_button.setVisible(True)
        if hasattr(self.window, "groupBox"):
            self.window.groupBox.setTitle("Local models")
            self.window.groupBox.setVisible(False)
        if hasattr(self.window, "groupBox_2"):
            self.window.groupBox_2.setTitle("Hub models")
            self.window.groupBox_2.setVisible(False)

        ignore_frame = QFrame(header)
        ignore_layout = QGridLayout(ignore_frame)
        self._configure_compact_form_grid(ignore_layout)

        for widget in (
            self.window.label_124,
            self.window.overview_ignore_before_input,
            self.window.label_125,
            self.window.overview_ignore_after_input,
            self.window.label_126,
        ):
            widget.setParent(ignore_frame)

        self.window.label_124.setText("Before")
        self.window.label_124.setProperty("fieldLabel", True)
        self.window.label_125.hide()
        self.window.label_126.hide()
        self.window.overview_ignore_before_input.setMaximumWidth(48)
        self.window.overview_ignore_after_input.setMaximumWidth(48)
        before_field = self._create_compact_field_block(
            "",
            self.window.overview_ignore_before_input,
            "sec",
            self.window.label_124,
            option_id="classifier_ignore_before",
            density="standard",
        )
        after_label = QLabel("After")
        after_label.setProperty("fieldLabel", True)
        after_field = self._create_compact_field_block(
            "",
            self.window.overview_ignore_after_input,
            "sec",
            after_label,
            option_id="classifier_ignore_after",
            density="standard",
        )
        self._add_compact_field_to_grid(ignore_layout, 0, 0, before_field)
        self._add_compact_field_to_grid(ignore_layout, 0, 1, after_field)
        self.window.classifier_ignore_frame = ignore_frame

        preset_field = self._create_compact_control_row(
            "Preset",
            self.window.classifier_mode_combo,
            label_widget=label,
            action_widgets=[self.window.classifier_run_button],
            option_id="classifier_preset",
            density="essential",
        )
        header_layout.addWidget(preset_field)
        header_layout.addWidget(ignore_frame)

        classifier_layout.addWidget(header, 0, 0)
        self._apply_section_visibility("CLASSIFICATION")

    def _create_waveform_placeholder(self):
        placeholder = QFrame(self.window.widget)
        placeholder.setObjectName("WaveformPlaceholder")
        placeholder.setMinimumWidth(420)
        placeholder.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QVBoxLayout(placeholder)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(6)

        title = QLabel("Open a recording to begin")
        title.setProperty("sectionTitle", True)
        subtitle = QLabel("The full workspace is already loaded. Open EDF, BrainVision, or FIF data, or restore a saved session to populate the waveform and results.")
        subtitle.setProperty("helperText", True)
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        hint = QLabel("Signal prep, detection, classification, and results controls stay available in the right panel.")
        hint.setProperty("helperText", True)
        hint.setWordWrap(True)
        layout.addWidget(hint)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        self.window.empty_open_button = QPushButton("Open Recording")
        self.window.empty_load_session_button = QPushButton("Load Session")
        self.window.empty_quick_button = QPushButton("Quick Detection")
        button_row.addWidget(self.window.empty_open_button)
        button_row.addWidget(self.window.empty_load_session_button)
        button_row.addWidget(self.window.empty_quick_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.window.waveform_placeholder = placeholder
        self._placeholder_target = self.window.widget
        self.window.widget.installEventFilter(self)
        placeholder.raise_()
        self._reposition_waveform_placeholder()

    def _get_waveform_placeholder_anchor_rect(self):
        graphics_widget = getattr(self.window, "waveform_graphics_widget", None)
        if graphics_widget is not None and graphics_widget.parent() is self.window.widget and graphics_widget.isVisible():
            return graphics_widget.geometry().adjusted(12, 10, -12, -10)

        target = getattr(self, "_placeholder_target", None)
        if target is None:
            return None
        return target.contentsRect().adjusted(20, 16, -20, -16)

    def _build_inspector_dock(self):
        density = self.ui_density
        dock = QDockWidget("Case Inspector", self.window)
        dock.setObjectName("caseInspectorDock")
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        dock.setTitleBarWidget(QWidget(dock))

        content = QWidget(dock)
        content.setObjectName("InspectorContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        hidden_state = self._build_active_run_group()
        hidden_state.hide()
        self.window.inspector_hidden_state_widget = hidden_state

        layout.addWidget(self._build_run_toolbar())
        layout.addWidget(self._build_results_section())
        self._install_inspector_tabs()

        self.window.tabWidget.setDocumentMode(True)
        self.window.tabWidget.setCurrentIndex(0)
        self.window.tabWidget.setMinimumWidth(0)
        self.window.tabWidget.setProperty("inspectorTabs", True)
        layout.addWidget(self.window.tabWidget, 1)

        dock.setWidget(content)
        self.window.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.window.inspector_dock = dock
        self.window.inspector_container = content
        self.window.inspector_scroll_area = None
        dock.setMinimumWidth(density.inspector_dock_min_width)
        dock.setMaximumWidth(density.inspector_dock_max_width)

    def _build_inspector_header(self):
        frame = QFrame(self.window)
        frame.setProperty("inspectorHeader", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(2, 0, 2, 9)
        layout.setSpacing(3)

        eyebrow = QLabel("CASE INSPECTOR", frame)
        eyebrow.setProperty("inspectorEyebrow", True)
        title = QLabel("Case Inspector", frame)
        title.setProperty("inspectorDockTitle", True)
        subtitle = QLabel(
            "Keep signal prep, detector passes, and the accepted run together while the waveform stays in view.",
            frame,
        )
        subtitle.setProperty("inspectorBodyText", True)
        subtitle.setWordWrap(True)

        layout.addWidget(eyebrow)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return frame

    def _get_inspector_section_subtitle(self, title):
        return ""

    def _wrap_inspector_section(self, title, child_widget):
        frame = QFrame(self.window.inspector_container)
        frame.setObjectName("InspectorSection")
        frame.setProperty("inspectorCard", True)
        section_layout = QVBoxLayout(frame)
        section_layout.setContentsMargins(5, 5, 5, 5)
        section_layout.setSpacing(3)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        title_label = QLabel(title.title(), frame)
        title_label.setProperty("inspectorSectionTitle", True)
        header_layout.addWidget(title_label, 0, Qt.AlignVCenter)
        header_layout.addStretch(1)
        header_layout.addWidget(self._create_section_settings_button(title, frame), 0, Qt.AlignTop)
        section_layout.addLayout(header_layout)
        rule = QFrame(frame)
        rule.setProperty("inspectorRule", True)
        section_layout.addWidget(rule)

        child_widget.setParent(frame)
        child_widget.setProperty("inspectorSectionBody", True)
        child_widget.show()
        section_layout.addWidget(child_widget)
        self._inspector_section_roots[title] = child_widget
        return frame

    def _build_single_page_inspector(self):
        if hasattr(self.window, "inspector_single_page"):
            return

        container = getattr(self.window, "inspector_container", None)
        if container is None:
            return
        layout = container.layout()
        if layout is None:
            return

        if hasattr(self.window, "tabWidget"):
            layout.removeWidget(self.window.tabWidget)
            self.window.tabWidget.hide()

        single_page = QWidget(container)
        single_page.setObjectName("InspectorSinglePage")
        single_layout = QVBoxLayout(single_page)
        single_layout.setContentsMargins(0, 0, 0, 0)
        single_layout.setSpacing(8)

        prepare_body = getattr(self.window, "prepare_tab_compact_container", None)
        if prepare_body is not None:
            prepare_body.setParent(single_page)
            single_layout.addWidget(self._wrap_inspector_section("SIGNAL", prepare_body))

        if hasattr(self.window, "detector_tab"):
            detector_layout = self.window.detector_tab.layout()
            if detector_layout is not None:
                detector_layout.setContentsMargins(0, 0, 0, 0)
                detector_layout.setVerticalSpacing(4)
            if hasattr(self.window, "detector_subtabs"):
                self.window.detector_subtabs.tabBar().hide()
            self.window.detector_tab.setParent(single_page)
            single_layout.addWidget(self._wrap_inspector_section("DETECTION", self.window.detector_tab))

        if hasattr(self.window, "classifier_tab"):
            classifier_layout = self.window.classifier_tab.layout()
            if classifier_layout is not None:
                classifier_layout.setContentsMargins(0, 0, 0, 0)
                classifier_layout.setVerticalSpacing(4)
            self._compact_classifier_panel()
            self.window.classifier_tab.setParent(single_page)
            single_layout.addWidget(self._wrap_inspector_section("CLASSIFICATION", self.window.classifier_tab))

        outputs_body = getattr(self.window, "prepare_outputs_container", None)
        if outputs_body is not None:
            outputs_body.hide()

        single_layout.addStretch(1)
        layout.addWidget(single_page, 1)
        self.window.inspector_single_page = single_page
        self._apply_all_section_visibility()

    def _make_summary_value_label(self, default_text="--"):
        value = QLabel(default_text)
        value.setProperty("metricValue", True)
        return value

    def _build_status_strip(self):
        status_bar = self.window.statusBar()
        status_bar.setSizeGripEnabled(False)

        content = QWidget(status_bar)
        layout = QHBoxLayout(content)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        self.window.activity_level_dot = QFrame(content)
        self.window.activity_level_dot.setFixedSize(10, 10)
        self.window.activity_level_dot.setProperty("statusDot", True)
        self.window.activity_level_dot.setProperty("activityLevel", "info")
        layout.addWidget(self.window.activity_level_dot, 0, Qt.AlignVCenter)

        self.window.activity_summary_label = QLabel("Ready")
        self.window.activity_summary_label.setProperty("helperText", True)
        self.window.activity_summary_label.setWordWrap(False)
        self.window.activity_summary_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self.window.activity_summary_label, 1)

        self.window.STDTextEdit.setParent(self.window)
        self.window.STDTextEdit.setProperty("consolePanel", True)
        self.window.STDTextEdit.setVisible(False)

        status_bar.addPermanentWidget(content, 1)
        self.window.status_strip = content

    def _clear_activity_panels(self):
        if hasattr(self.window, "STDTextEdit"):
            self.window.STDTextEdit.clear()
        if hasattr(self.window, "activity_summary_label"):
            self.window.activity_summary_label.setText("Ready")
        if hasattr(self.window, "activity_level_dot"):
            set_dynamic_property(self.window.activity_level_dot, "activityLevel", "info")

    def _build_run_toolbar(self):
        frame = QFrame(self.window)
        frame.setProperty("inspectorCard", True)
        self.window.run_actions_card = frame
        outer_layout = QVBoxLayout(frame)
        outer_layout.setContentsMargins(5, 5, 5, 5)
        outer_layout.setSpacing(3)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(4)
        title = QLabel("Run Actions", frame)
        title.setProperty("inspectorCardHeading", True)
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(self._create_section_settings_button("RUN_ACTIONS", frame))
        outer_layout.addLayout(title_row)
        rule = QFrame(frame)
        rule.setProperty("inspectorRule", True)
        outer_layout.addWidget(rule)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)

        def make_toolbar_label(text, parent):
            label = QLabel(text, parent)
            label.setProperty("toolbarLabel", True)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            label.setMinimumWidth(46)
            label.setMaximumWidth(46)
            return label

        self.window.active_run_selector = QComboBox(frame)
        self.window.active_run_selector.setToolTip("Choose the active run to review")
        self.window.active_run_selector.setEnabled(False)
        self.window.active_run_selector.setMinimumContentsLength(14)
        self.window.active_run_selector.setMinimumWidth(124)
        self._apply_compact_input_height(self.window.active_run_selector)
        self.window.active_run_selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        active_row_frame = QFrame(frame)
        active_row = QHBoxLayout(active_row_frame)
        active_row.setContentsMargins(0, 0, 0, 0)
        active_row.setSpacing(3)
        active_row.addWidget(make_toolbar_label("Active", active_row_frame))
        active_row.addWidget(self.window.active_run_selector, 1)
        grid.addWidget(
            self._mark_section_option(active_row_frame, "run_active_selector", "Active run", "essential"),
            0,
            0,
            1,
            3,
        )

        self.window.accept_run_button = QPushButton("Accept")
        self.window.compare_runs_button = QPushButton("Run Stats")
        self.window.accept_run_button.setToolTip("Mark the selected run as accepted")
        self.window.compare_runs_button.setToolTip("Open run statistics, channel ranking, and detector agreement for the current workflow")
        self.window.accept_run_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.window.compare_runs_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.window.accept_run_button.setMinimumWidth(max(54, self.window.accept_run_button.sizeHint().width()))
        self.window.compare_runs_button.setMinimumWidth(max(76, self.window.compare_runs_button.sizeHint().width()))
        self.window.accept_run_button.setProperty("inspectorPrimary", True)
        self.window.compare_runs_button.setProperty("inspectorSecondary", True)
        self.window.accept_run_button.setEnabled(False)
        self.window.compare_runs_button.setEnabled(False)
        action_row_frame = QFrame(frame)
        action_row = QHBoxLayout(action_row_frame)
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(4)
        action_row.addWidget(self.window.accept_run_button, 1)
        action_row.addWidget(self.window.compare_runs_button, 1)
        grid.addWidget(
            self._mark_section_option(action_row_frame, "run_decision_actions", "Decision buttons", "standard"),
            1,
            0,
            1,
            3,
        )

        self.window.combo_box_biomarker.setParent(frame)
        self.window.combo_box_biomarker.setMinimumWidth(96)
        self._apply_compact_input_height(self.window.combo_box_biomarker)
        self.window.combo_box_biomarker.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.window.combo_box_biomarker.setToolTip("Choose the biomarker workflow for the next run")
        next_row_frame = QFrame(frame)
        next_row = QHBoxLayout(next_row_frame)
        next_row.setContentsMargins(0, 0, 0, 0)
        next_row.setSpacing(3)
        next_row.addWidget(make_toolbar_label("Next", next_row_frame))
        next_row.addWidget(self.window.combo_box_biomarker, 1)
        grid.addWidget(
            self._mark_section_option(next_row_frame, "run_next_creator", "Next workflow", "standard"),
            2,
            0,
            1,
            2,
        )

        self.window.new_run_button = QToolButton(frame)
        self.window.new_run_button.setText("New")
        self.window.new_run_button.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.window.new_run_button)
        self.window.new_hfo_run_action = menu.addAction("New HFO Run")
        self.window.new_spindle_run_action = menu.addAction("New Spindle Run")
        self.window.new_spike_run_action = menu.addAction("New Spike Run")
        self.window.new_run_button.setMenu(menu)
        self.window.new_run_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.window.new_run_button.setAutoRaise(False)
        self.window.new_run_button.setMinimumWidth(56)
        self.window.new_run_button.setToolTip("Create a new run for the selected workflow")
        self.window.new_run_button.setProperty("runLauncher", True)
        self.window.new_run_button.setFixedHeight(self.ui_density.compact_input_height)
        next_action_frame = QFrame(frame)
        next_action = QHBoxLayout(next_action_frame)
        next_action.setContentsMargins(0, 0, 0, 0)
        next_action.setSpacing(0)
        next_action.addWidget(self.window.new_run_button)
        grid.addWidget(
            self._mark_section_option(next_action_frame, "run_next_creator", "Next workflow", "standard"),
            2,
            2,
        )

        self.window.switch_run_button = QPushButton("Activate")
        self.window.switch_run_button.setVisible(False)
        self.window.switch_run_button.setEnabled(False)

        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(2, 92)
        outer_layout.addLayout(grid)
        self._inspector_section_roots["RUN_ACTIONS"] = frame
        self._apply_section_visibility("RUN_ACTIONS")
        return frame

    def _build_results_section(self):
        frame = QFrame(self.window)
        frame.setProperty("inspectorCard", True)
        self.window.results_card = frame
        outer_layout = QVBoxLayout(frame)
        outer_layout.setContentsMargins(5, 5, 5, 5)
        outer_layout.setSpacing(3)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(4)
        title = QLabel("Results", frame)
        title.setProperty("inspectorCardHeading", True)
        title_row.addWidget(title)
        title_row.addStretch(1)
        outer_layout.addLayout(title_row)

        rule = QFrame(frame)
        rule.setProperty("inspectorRule", True)
        outer_layout.addWidget(rule)

        self.window.annotation_button.setParent(frame)
        self.window.annotation_button.setText("Open Review")
        self.window.annotation_button.setToolTip("Open the event-by-event review window for the active run")
        self.window.annotation_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.window.annotation_button.setMinimumWidth(0)
        self.window.annotation_button.setProperty("inspectorSecondary", True)
        self.window.annotation_button.setVisible(True)

        self.window.save_npz_button.setParent(frame)
        self.window.save_npz_button.setText("Save Session")
        self.window.save_npz_button.setToolTip("Save the current PyBrain session")
        self.window.save_npz_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.window.save_npz_button.setMinimumWidth(0)
        self.window.save_npz_button.setProperty("inspectorSecondary", True)
        self.window.save_npz_button.setVisible(True)

        self.window.save_csv_button.setParent(frame)
        self.window.save_csv_button.setText("Export Workbook")
        self.window.save_csv_button.setToolTip("Export the current accepted result workbook")
        self.window.save_csv_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.window.save_csv_button.setMinimumWidth(0)
        self.window.save_csv_button.setProperty("inspectorSecondary", True)
        self.window.save_csv_button.setVisible(True)

        button_grid = QGridLayout()
        button_grid.setContentsMargins(0, 0, 0, 0)
        button_grid.setHorizontalSpacing(4)
        button_grid.setVerticalSpacing(4)
        button_grid.addWidget(self.window.annotation_button, 0, 0)
        button_grid.addWidget(self.window.save_npz_button, 0, 1)
        button_grid.addWidget(self.window.save_csv_button, 0, 2)
        button_grid.setColumnStretch(0, 1)
        button_grid.setColumnStretch(1, 1)
        button_grid.setColumnStretch(2, 1)
        outer_layout.addLayout(button_grid)

        return frame

    def _create_compact_field_block(self, label_text, input_widget, unit_text="", label_widget=None, option_id=None, density="standard"):
        frame = QFrame()
        frame.setObjectName("CompactField")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        raw_label_text = label_widget.text() if label_widget is not None and label_widget.text() else label_text
        label = label_widget or QLabel(raw_label_text)
        label.setProperty("fieldLabel", True)
        label.setText(self._compact_label_text(raw_label_text))
        label.setFixedWidth(self._compact_label_width())
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        layout.addWidget(label)

        self._apply_compact_input_height(input_widget)
        input_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(input_widget, 1)
        input_widget.show()
        if unit_text:
            unit = QLabel(unit_text)
            unit.setProperty("fieldUnit", True)
            unit.setFixedWidth(self._compact_unit_width())
            layout.addWidget(unit, 0, Qt.AlignRight | Qt.AlignVCenter)
        return self._mark_section_option(frame, option_id or self._normalize_option_id(raw_label_text), raw_label_text, density)

    def _create_compact_range_field_block(self, label_text, low_widget, high_widget, unit_text="", label_widget=None, option_id=None, density="standard"):
        frame = QFrame()
        frame.setObjectName("CompactField")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        raw_label_text = label_widget.text() if label_widget is not None and label_widget.text() else label_text
        label = label_widget or QLabel(raw_label_text)
        label.setProperty("fieldLabel", True)
        label.setText(self._compact_label_text(raw_label_text))
        label.setFixedWidth(self._compact_label_width())
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        layout.addWidget(label)

        for widget in (low_widget, high_widget):
            self._apply_compact_input_height(widget)
            widget.setAlignment(Qt.AlignCenter)
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        dash = QLabel("to")
        dash.setProperty("fieldUnit", True)
        dash.setFixedWidth(self._compact_unit_width())
        layout.addWidget(low_widget, 1)
        layout.addWidget(dash, 0, Qt.AlignCenter)
        layout.addWidget(high_widget, 1)
        low_widget.show()
        high_widget.show()
        if unit_text:
            unit = QLabel(unit_text)
            unit.setProperty("fieldUnit", True)
            unit.setFixedWidth(self._compact_unit_width())
            layout.addWidget(unit, 0, Qt.AlignRight | Qt.AlignVCenter)
        return self._mark_section_option(frame, option_id or self._normalize_option_id(raw_label_text), raw_label_text, density)

    def _rebuild_filter_section(self):
        groupbox = getattr(self.window, "filters_groupbox", None)
        if groupbox is None or getattr(self.window, "_filter_section_rebuilt", False):
            return
        layout = getattr(groupbox, "layout", lambda: None)()
        if layout is None:
            return

        groupbox.setTitle("")
        # The filter widgets come from the loaded .ui file and are reused below, so
        # we must detach them from the old layout instead of deleting them.
        detach_layout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(4)

        field_defs = [
            ("Low cutoff", self.window.fp_input, "signal_low_cutoff", "essential"),
            ("High cutoff", self.window.fs_input, "signal_high_cutoff", "essential"),
            ("Pass ripple", self.window.rp_input, "signal_pass_ripple", "advanced"),
            ("Stop atten.", self.window.rs_input, "signal_stop_atten", "advanced"),
        ]
        for _title, widget, _option_id, _density in field_defs:
            widget.setMaximumWidth(120)
            widget.setAlignment(Qt.AlignCenter)

        field_grid = QGridLayout()
        self._configure_compact_form_grid(field_grid, action_column=True)
        low_field = self._create_compact_field_block(
            "Low cutoff",
            self.window.fp_input,
            option_id="signal_low_cutoff",
            density="essential",
        )
        high_field = self._create_compact_field_block(
            "High cutoff",
            self.window.fs_input,
            option_id="signal_high_cutoff",
            density="essential",
        )
        field_grid.addWidget(low_field, 0, 0)
        field_grid.addWidget(high_field, 0, 1)
        self.window.overview_filter_button.setText("Filter")
        self.window.overview_filter_button.setMinimumWidth(68)
        self.window.overview_filter_button.setProperty("inspectorPrimary", True)
        field_grid.addWidget(self.window.overview_filter_button, 0, 2)
        self.window.overview_filter_button.show()
        layout.addLayout(field_grid, 0, 0, 1, 2)

        advanced_grid = QGridLayout()
        self._configure_compact_form_grid(advanced_grid)
        self._add_compact_field_to_grid(
            advanced_grid,
            0,
            0,
            self._create_compact_field_block(
                "Pass ripple",
                self.window.rp_input,
                option_id="signal_pass_ripple",
                density="advanced",
            ),
        )
        self._add_compact_field_to_grid(
            advanced_grid,
            0,
            1,
            self._create_compact_field_block(
                "Stop atten.",
                self.window.rs_input,
                option_id="signal_stop_atten",
                density="advanced",
            ),
        )
        layout.addLayout(advanced_grid, 1, 0, 1, 2)

        action_row = QGridLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setHorizontalSpacing(4)
        action_row.setVerticalSpacing(4)
        if hasattr(self.window, "prepare_runtime_frame"):
            self.window.prepare_runtime_frame.setParent(groupbox)
            self.window.prepare_runtime_frame.show()
            workers_frame = self._mark_section_option(self.window.prepare_runtime_frame, "signal_workers", "Workers", "advanced")
            action_row.addWidget(workers_frame, 0, 0)
            action_row.setColumnStretch(0, 1)
        layout.addLayout(action_row, 2, 0, 1, 2)
        self.window._filter_section_rebuilt = True
        self._apply_section_visibility("SIGNAL")

    def _compact_classifier_panel(self):
        grid = getattr(self.window, "gridLayout_19", None)
        if grid is None or getattr(self.window, "_classifier_panel_compacted", False):
            return

        grid.setContentsMargins(0, 0, 0, 0)
        self._configure_compact_form_grid(grid)

        for spacer_row in (1, 3, 11):
            item = grid.itemAtPosition(spacer_row, 0)
            if item is not None and item.spacerItem() is not None:
                grid.removeItem(item)
            grid.setRowMinimumHeight(spacer_row, 0)
            grid.setRowStretch(spacer_row, 0)

        for widget_name in ("groupBox", "groupBox_2"):
            widget = getattr(self.window, widget_name, None)
            if widget is not None:
                grid.removeWidget(widget)
                widget.hide()

        custom_sources_frame = QFrame(self.window.classifier_tab)
        custom_sources_layout = QVBoxLayout(custom_sources_frame)
        custom_sources_layout.setContentsMargins(0, 0, 0, 0)
        custom_sources_layout.setSpacing(4)
        for widget_name in ("groupBox", "groupBox_2"):
            widget = getattr(self.window, widget_name, None)
            if widget is not None:
                widget.setParent(custom_sources_frame)
                custom_sources_layout.addWidget(widget)
        custom_sources_frame.hide()
        self.window.classifier_custom_sources_frame = custom_sources_frame

        removable_rows = (4, 7, 8, 9, 10)
        for row in removable_rows:
            item = grid.itemAtPosition(row, 0)
            if item is not None:
                grid.removeItem(item)

        self.window.label_70.setParent(self.window.classifier_tab)
        self.window.classifier_device_input.setParent(self.window.classifier_tab)
        self.window.label_27.setParent(self.window.classifier_tab)
        self.window.classifier_batch_size_input.setParent(self.window.classifier_tab)
        self.window.use_spike_checkbox.setParent(self.window.classifier_tab)
        self.window.use_ehfo_checkbox.setParent(self.window.classifier_tab)

        device_field = self._create_compact_field_block(
            "",
            self.window.classifier_device_input,
            label_widget=self.window.label_70,
            option_id="classifier_device",
            density="advanced",
        )
        batch_field = self._create_compact_field_block(
            "",
            self.window.classifier_batch_size_input,
            label_widget=self.window.label_27,
            option_id="classifier_batch_size",
            density="advanced",
        )

        toggle_frame = QFrame(self.window.classifier_tab)
        toggle_layout = QHBoxLayout(toggle_frame)
        toggle_layout.setContentsMargins(self._compact_label_width() + 3, 0, 0, 0)
        toggle_layout.setSpacing(16)
        for toggle in (self.window.use_spike_checkbox, self.window.use_ehfo_checkbox):
            toggle.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        toggle_layout.addWidget(self.window.use_spike_checkbox, 0, Qt.AlignLeft)
        toggle_layout.addWidget(self.window.use_ehfo_checkbox, 0, Qt.AlignLeft)
        toggle_layout.addStretch(1)

        apply_frame = QFrame(self.window.classifier_tab)
        apply_layout = QHBoxLayout(apply_frame)
        apply_layout.setContentsMargins(0, 0, 0, 0)
        apply_layout.setSpacing(0)
        apply_layout.addStretch(1)
        self.window.classifier_apply_button.setParent(apply_frame)
        apply_layout.addWidget(self.window.classifier_apply_button, 0, Qt.AlignRight)
        apply_frame.hide()
        self.window.classifier_custom_apply_frame = apply_frame

        self._add_compact_field_to_grid(grid, 0, 0, device_field)
        self._add_compact_field_to_grid(grid, 0, 1, batch_field)
        grid.addWidget(self._mark_section_option(toggle_frame, "classifier_toggles", "Classifier toggles", "essential"), 1, 0, 1, 6)
        grid.addWidget(self._mark_section_option(custom_sources_frame, "classifier_custom_sources", "Custom model sources", "advanced"), 2, 0, 1, 6)
        grid.addWidget(self._mark_section_option(apply_frame, "classifier_custom_apply", "Apply classifier settings", "standard"), 3, 0, 1, 6)

        self.window._classifier_panel_compacted = True
        self._apply_section_visibility("CLASSIFICATION")

    def _build_active_run_group(self):
        box = QFrame(self.window)
        box.setProperty("inspectorHero", True)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        eyebrow = QLabel("SESSION SNAPSHOT", box)
        eyebrow.setProperty("inspectorEyebrow", True)
        layout.addWidget(eyebrow)

        self.window.analysis_summary_label = QLabel("No saved runs yet", box)
        self.window.analysis_summary_label.setProperty("inspectorBodyText", True)
        self.window.analysis_summary_label.setWordWrap(True)
        layout.addWidget(self.window.analysis_summary_label)

        hidden_summary_container = QWidget(box)
        hidden_summary_layout = QHBoxLayout(hidden_summary_container)
        hidden_summary_layout.setContentsMargins(0, 0, 0, 0)
        hidden_summary_layout.setSpacing(0)
        summary_fields = (
            "decision_runs_value",
            "decision_active_value",
            "decision_accepted_value",
            "decision_channel_value",
        )
        for attr_name in summary_fields:
            value = self._make_summary_value_label()
            value.setVisible(False)
            setattr(self.window, attr_name, value)
            hidden_summary_layout.addWidget(value)
        hidden_summary_container.setVisible(False)
        layout.addWidget(hidden_summary_container)

        self.window.active_run_name_label = QLabel("No active run", box)
        self.window.active_run_name_label.setProperty("inspectorHeroTitle", True)
        layout.addWidget(self.window.active_run_name_label)

        self.window.active_run_meta_label = QLabel("No run yet", box)
        self.window.active_run_meta_label.setProperty("inspectorHeroMeta", True)
        self.window.active_run_meta_label.setWordWrap(True)
        layout.addWidget(self.window.active_run_meta_label)

        self.window.active_run_status_label = QLabel("Status --", box)
        self.window.active_detector_label = QLabel("Detector --", box)
        self.window.active_classifier_label = QLabel("Classifier --", box)

        chip_grid = QGridLayout()
        chip_grid.setContentsMargins(0, 0, 0, 0)
        chip_grid.setHorizontalSpacing(8)
        chip_grid.setVerticalSpacing(8)
        chip_grid.addWidget(self._create_inspector_meta_chip("State", self.window.active_run_status_label, box), 0, 0)
        chip_grid.addWidget(self._create_inspector_meta_chip("Detector", self.window.active_detector_label, box), 0, 1)
        chip_grid.addWidget(self._create_inspector_meta_chip("Classifier", self.window.active_classifier_label, box), 1, 0, 1, 2)
        chip_grid.setColumnStretch(0, 1)
        chip_grid.setColumnStretch(1, 1)
        layout.addLayout(chip_grid)

        detector_heading = QLabel("Current detector setup", box)
        detector_heading.setProperty("inspectorCardHeading", True)
        layout.addWidget(detector_heading)

        self.window.active_run_param_table = QTableWidget(0, 2, box)
        self.window.active_run_param_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.window.active_run_param_table.verticalHeader().setVisible(False)
        self.window.active_run_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.window.active_run_param_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.window.active_run_param_table.setFocusPolicy(Qt.NoFocus)
        self.window.active_run_param_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.window.active_run_param_table.setShowGrid(False)
        self.window.active_run_param_table.setMinimumHeight(84)
        self.window.active_run_param_table.setMaximumHeight(108)
        self.window.active_run_param_table.setProperty("summaryTable", True)
        self.window.active_run_param_table.verticalHeader().setDefaultSectionSize(18)
        self.window.active_run_param_table.horizontalHeader().setStretchLastSection(True)
        self.window.active_run_param_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.window.active_run_param_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.window.active_run_param_table)

        classifier_heading = QLabel("Classifier settings", box)
        classifier_heading.setProperty("inspectorCardHeading", True)
        layout.addWidget(classifier_heading)

        self.window.active_classifier_table = QTableWidget(0, 2, box)
        self.window.active_classifier_table.setHorizontalHeaderLabels(["Setting", "Value"])
        self.window.active_classifier_table.verticalHeader().setVisible(False)
        self.window.active_classifier_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.window.active_classifier_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.window.active_classifier_table.setFocusPolicy(Qt.NoFocus)
        self.window.active_classifier_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.window.active_classifier_table.setShowGrid(False)
        self.window.active_classifier_table.setMinimumHeight(52)
        self.window.active_classifier_table.setMaximumHeight(70)
        self.window.active_classifier_table.setProperty("summaryTable", True)
        self.window.active_classifier_table.verticalHeader().setDefaultSectionSize(18)
        self.window.active_classifier_table.horizontalHeader().setStretchLastSection(True)
        self.window.active_classifier_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.window.active_classifier_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.window.active_classifier_table)
        return box

    def _create_inspector_meta_chip(self, title, value_label, parent):
        chip = QFrame(parent)
        chip.setProperty("inspectorChip", True)
        layout = QVBoxLayout(chip)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        title_label = QLabel(title, chip)
        title_label.setProperty("inspectorChipLabel", True)
        layout.addWidget(title_label)

        value_label.setParent(chip)
        value_label.setProperty("inspectorChipValue", True)
        value_label.setWordWrap(True)
        value_label.show()
        layout.addWidget(value_label)
        return chip

    def _build_run_management_group(self):
        box = QGroupBox("Runs")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.window.run_summary_label = QLabel("No runs yet.")
        self.window.run_summary_label.setProperty("helperText", True)
        self.window.run_summary_label.setWordWrap(False)
        layout.addWidget(self.window.run_summary_label)

        self.window.run_table = QTableWidget(0, 5)
        self.window.run_table.setHorizontalHeaderLabels(["Vis", "Bio", "Detector", "Events", "State"])
        self.window.run_table.verticalHeader().setVisible(False)
        self.window.run_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.window.run_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.window.run_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.window.run_table.horizontalHeader().setStretchLastSection(False)
        self.window.run_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.window.run_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.window.run_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.window.run_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.window.run_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.window.run_table.setMinimumHeight(88)
        self.window.run_table.setMaximumHeight(136)
        self.window.run_table.setProperty("summaryTable", True)
        layout.addWidget(self.window.run_table)

        channel_title = QLabel("Channel Ranking")
        channel_title.setProperty("sectionTitle", True)
        layout.addWidget(channel_title)

        self.window.channel_table = QTableWidget(0, 4)
        self.window.channel_table.setHorizontalHeaderLabels(["Channel", "Acc", "All", "Rank"])
        self.window.channel_table.verticalHeader().setVisible(False)
        self.window.channel_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.window.channel_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.window.channel_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.window.channel_table.horizontalHeader().setStretchLastSection(False)
        self.window.channel_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.window.channel_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.window.channel_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.window.channel_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.window.channel_table.setMinimumHeight(76)
        self.window.channel_table.setMaximumHeight(96)
        self.window.channel_table.setProperty("summaryTable", True)
        layout.addWidget(self.window.channel_table)

        compare_title = QLabel("Run Overlap")
        compare_title.setProperty("sectionTitle", True)
        layout.addWidget(compare_title)

        self.window.comparison_table = QTableWidget(0, 5)
        self.window.comparison_table.setHorizontalHeaderLabels(["A", "B", "Overlap", "Only", "Jaccard"])
        self.window.comparison_table.verticalHeader().setVisible(False)
        self.window.comparison_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.window.comparison_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.window.comparison_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.window.comparison_table.horizontalHeader().setStretchLastSection(False)
        self.window.comparison_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.window.comparison_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.window.comparison_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.window.comparison_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.window.comparison_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.window.comparison_table.setMinimumHeight(64)
        self.window.comparison_table.setMaximumHeight(82)
        self.window.comparison_table.setProperty("summaryTable", True)
        layout.addWidget(self.window.comparison_table)
        return box

    def _build_run_statistics_dialog(self):
        dialog = QDialog(self.window)
        dialog.setWindowTitle("Run Statistics")
        dialog.resize(860, 680)
        dialog.setModal(False)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.window.run_stats_activate_button = QPushButton("Activate Selected")
        self.window.run_stats_accept_button = QPushButton("Accept Selected")
        self.window.run_stats_export_button = QPushButton("Export Workbook")
        self.window.run_stats_report_button = QPushButton("Export Report")
        action_row.addWidget(self.window.run_stats_activate_button)
        action_row.addWidget(self.window.run_stats_accept_button)
        action_row.addWidget(self.window.run_stats_export_button)
        action_row.addWidget(self.window.run_stats_report_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        layout.addWidget(self._build_run_management_group(), 1)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        close_row.addWidget(close_button)
        layout.addLayout(close_row)

        apply_subwindow_theme(dialog)
        self.window.run_stats_dialog = dialog

    def _install_inspector_tabs(self):
        if getattr(self.window, "runs_tab_installed", False):
            return
        self.window.tabWidget.setTabText(0, "Prepare")
        self.window.tabWidget.setTabText(1, "Detect")
        self.window.tabWidget.setTabText(2, "Classify")
        self.window.runs_tab_installed = True

    def _make_waveform_tool_button(self, *, icon=None, text="", tooltip="", checkable=False, icon_only=False):
        button = QToolButton(self.window.widget)
        button.setProperty("waveformTool", True)
        button.setProperty("waveformLabeledTool", not icon_only)
        button.setAutoRaise(False)
        button.setCursor(Qt.PointingHandCursor)
        button.setText(text)
        if text:
            button.setAccessibleName(text)
        if icon is not None:
            button.setIcon(icon)
            button.setIconSize(QtCore.QSize(16, 16))
            button.setToolButtonStyle(Qt.ToolButtonIconOnly if icon_only else Qt.ToolButtonTextBesideIcon)
        else:
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        if tooltip:
            button.setToolTip(tooltip)
            button.setStatusTip(tooltip)
        button.setCheckable(checkable)
        return button

    def _make_waveform_preset_button(self, *, text="", tooltip="", checkable=True):
        button = QToolButton(self.window.widget)
        button.setProperty("waveformPreset", True)
        button.setAutoRaise(False)
        button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        button.setText(text)
        if tooltip:
            button.setToolTip(tooltip)
            button.setStatusTip(tooltip)
        button.setCheckable(checkable)
        return button

    def _make_waveform_badge_button(self, *, text="", tooltip=""):
        button = QToolButton(self.window.widget)
        button.setProperty("waveformBadge", True)
        button.setProperty("preserveButtonHeight", True)
        button.setAutoRaise(False)
        button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        button.setText(text)
        button.setCursor(Qt.PointingHandCursor)
        if tooltip:
            button.setToolTip(tooltip)
            button.setStatusTip(tooltip)
        return button

    def _make_waveform_toolbar_group(self, parent, *, spacing=4):
        frame = QFrame(parent)
        frame.setProperty("waveformRow", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(spacing)
        return frame, layout

    def _make_waveform_toolbar_slot(self, parent, *, width, height=26):
        frame = QFrame(parent)
        frame.setProperty("waveformSlot", True)
        frame.setFixedSize(width, height)
        frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        return frame, layout

    def _mount_waveform_toolbar_slot(self, parent, widget, *, width, height=26):
        slot, layout = self._make_waveform_toolbar_slot(parent, width=width, height=height)
        widget.setFixedSize(width, height)
        layout.addWidget(widget)
        return slot

    def _lock_waveform_toolbar_button_widths(self, width=35):
        toolbar_buttons = (
            "prev_event_button",
            "center_event_button",
            "next_event_button",
            "pending_event_button",
            "open_review_button",
            "snapshot_button",
            "raw_tool_button",
            "normalize_tool_button",
            "filtered_tool_button",
            "filter60_tool_button",
            "review_channels_button",
            "referential_tool_button",
            "average_reference_button",
            "auto_bipolar_button",
            "highlight_channel_button",
            "neighbor_channels_button",
            "clean_view_button",
            "overlap_review_button",
            "montage_tool_button",
            "event_channels_button",
            "all_channels_button",
            "cursor_tool_button",
            "measure_tool_button",
            "hotspot_tool_button",
            "go_to_time_button",
            "zoom_out_button",
            "zoom_in_button",
        )
        for attr_name in toolbar_buttons:
            button = getattr(self.window, attr_name, None)
            if button is None:
                continue
            if button.property("waveformLabeledTool"):
                content_width = max(button.minimumSizeHint().width(), button.sizeHint().width())
                locked_width = max(width + 10, content_width + 2)
                locked_width = min(98, locked_width)
            else:
                locked_width = width
            button.setMinimumWidth(locked_width)
            button.setMaximumWidth(locked_width)
            button.setFixedWidth(locked_width)

    def _build_waveform_toolbar(self):
        if hasattr(self.window, "waveform_toolbar_frame"):
            return

        top_controls = self.window.horizontalLayout_11
        secondary_controls = self.window.horizontalLayout_14

        while top_controls.count():
            item = top_controls.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(self.window.widget_2)

        while secondary_controls.count():
            item = secondary_controls.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(self.window.waveformWiget)

        self.window.prev_event_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("prev"),
            text="Prev",
            tooltip="Jump to the previous detected event",
            icon_only=True,
        )
        self.window.center_event_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("center"),
            text="Center",
            tooltip="Center the waveform on the current event",
            icon_only=True,
        )
        self.window.next_event_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("next"),
            text="Next",
            tooltip="Jump to the next detected event",
            icon_only=True,
        )
        self.window.pending_event_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("pending"),
            text="Pending",
            tooltip="Jump to the next unreviewed detected event",
            icon_only=True,
        )
        self.window.open_review_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("review"),
            text="Review",
            tooltip="Open the detailed event review workspace for the active run",
            icon_only=True,
        )
        self.window.snapshot_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("snap"),
            text="Snap",
            tooltip="Save the current waveform view as an image",
            icon_only=True,
        )

        toolbar_item_height = 26
        toolbar_row_spacing = 3
        toolbar_row_count = 3
        toolbar_shell_height = max(
            (toolbar_item_height * toolbar_row_count) + (toolbar_row_spacing * (toolbar_row_count - 1)) + 16,
            self.ui_density.waveform_control_strip_height + 4,
        )
        slot_widths = {
            "window": 112,
            "step": 80,
            "amplitude": 112,
            "jump": 80,
            "visible_channels": 56,
            "event_position": 128,
            "montage_status": 88,
            "montage_break": 64,
            "measurement": 172,
        }

        self.window.event_position_label = QLabel("No events")
        self.window.event_position_label.setProperty("waveformBadge", True)
        self.window.event_position_label.setAlignment(Qt.AlignCenter)

        for widget in (
            self.window.normalize_vertical_input,
            self.window.toggle_filtered_checkbox,
            self.window.Filter60Button,
            self.window.bipolar_button,
            self.window.Choose_Channels_Button,
            self.window.waveform_plot_button,
        ):
            widget.hide()

        self.window.waveform_toolbar_frame = QFrame(self.window.widget_2)
        self.window.waveform_toolbar_frame.setProperty("metricCard", True)
        self.window.waveform_toolbar_frame.setProperty("waveformToolbarShell", True)
        self.window.waveform_toolbar_frame.setFixedHeight(toolbar_shell_height)
        self.window.waveform_toolbar_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.window.waveform_toolbar_frame.installEventFilter(self)
        self.window.widget_2.setMinimumHeight(toolbar_shell_height)
        self.window.widget_2.setMaximumHeight(toolbar_shell_height)
        toolbar_layout = QVBoxLayout(self.window.waveform_toolbar_frame)
        toolbar_layout.setContentsMargins(8, 6, 8, 6)
        toolbar_layout.setSpacing(toolbar_row_spacing)

        self.window.normalize_tool_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("normalize"),
            text="Norm",
            tooltip="Normalize each visible channel",
            checkable=True,
            icon_only=False,
        )
        self.window.raw_tool_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("raw"),
            text="Raw",
            tooltip="Show the raw waveform. This is the default source view.",
            checkable=True,
            icon_only=False,
        )
        self.window.filtered_tool_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("filtered"),
            text="Filt",
            tooltip="Show the filtered waveform. Click Raw or Reset View to return to the source trace.",
            checkable=True,
            icon_only=False,
        )
        self.window.filter60_tool_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("notch"),
            text="60 Hz",
            tooltip="Toggle the 60 Hz cleanup view",
            checkable=True,
            icon_only=False,
        )
        self.window.review_channels_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("channels"),
            text="Channels",
            tooltip="Open the channel selection workspace",
            icon_only=False,
        )
        self.window.overlap_review_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("overlap"),
            text="Overlap",
            tooltip="Keep the first overlap and tag or hide the later cross-channel duplicates",
            icon_only=False,
        )
        self.window.montage_tool_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("montage"),
            text="Montage",
            tooltip="Build or apply bipolar/montage channel views",
            icon_only=False,
        )
        self.window.referential_tool_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("referential"),
            text="Ref",
            tooltip="Return to the referential source channel view. This exits derived montage views.",
            checkable=True,
            icon_only=False,
        )
        self.window.average_reference_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("average"),
            text="Avg Ref",
            tooltip="Show average-reference derived channels. Click again, Ref, or Reset View to return.",
            checkable=True,
            icon_only=False,
        )
        self.window.auto_bipolar_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("auto_bipolar"),
            text="Auto Bp",
            tooltip="Automatically build conventional EEG bipolar chains or adjacent iEEG bipolar channels. Click again, Ref, or Reset View to return.",
            checkable=True,
            icon_only=False,
        )
        self.window.montage_status_badge = self._make_waveform_badge_button(
            tooltip="Show auto bipolar montage details",
        )
        self.window.montage_status_badge.setVisible(False)
        self.window.montage_break_badge = self._make_waveform_badge_button(
            tooltip="Show broken-chain details",
        )
        self.window.montage_break_badge.setVisible(False)
        self.window.highlight_channel_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("focus"),
            text="Highlight",
            tooltip="Highlight the selected channel without hiding the other visible channels. Click again or Reset View to exit.",
            checkable=True,
            icon_only=False,
        )
        self.window.neighbor_channels_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("adjacent"),
            text="Neighbors",
            tooltip="Focus the highlighted channel together with adjacent channels. Click again, All, or Reset View to return.",
            checkable=True,
            icon_only=False,
        )
        self.window.clean_view_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("clean"),
            text="Clean",
            tooltip="Hide explicitly bad or flat source channels. Click again, All, or Reset View to return.",
            checkable=True,
            icon_only=False,
        )
        self.window.event_channels_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("events"),
            text="Events",
            tooltip="Only show channels with detected events in the active run. Click again, All, or Reset View to return.",
            checkable=True,
            icon_only=False,
        )
        self.window.all_channels_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("all"),
            text="All",
            tooltip="Return to the full referential source channel list.",
            checkable=True,
            icon_only=False,
        )

        self.window.go_to_time_input = QDoubleSpinBox(self.window.waveform_toolbar_frame)
        self.window.go_to_time_input.setDecimals(2)
        self.window.go_to_time_input.setMinimum(0.0)
        self.window.go_to_time_input.setMaximum(0.0)
        self.window.go_to_time_input.setPrefix("Go ")
        self.window.go_to_time_input.setSuffix(" s")
        self.window.go_to_time_input.setAlignment(Qt.AlignCenter)
        self.window.go_to_time_input.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.window.go_to_time_input.setProperty("waveformField", True)
        self.window.go_to_time_input.setProperty("waveformStepper", False)
        self.window.go_to_time_input.setToolTip("Jump to a specific time in seconds")

        self.window.waveform_amplitude_input = QDoubleSpinBox(self.window.waveform_toolbar_frame)
        self.window.waveform_amplitude_input.setDecimals(2)
        self.window.waveform_amplitude_input.setSingleStep(0.25)
        self.window.waveform_amplitude_input.setMinimum(0.25)
        self.window.waveform_amplitude_input.setMaximum(10.0)
        self.window.waveform_amplitude_input.setValue(1.0)
        self.window.waveform_amplitude_input.setKeyboardTracking(False)
        self.window.waveform_amplitude_input.setPrefix("Amp ")
        self.window.waveform_amplitude_input.setSuffix("x")
        self.window.waveform_amplitude_input.setAlignment(Qt.AlignCenter)
        self.window.waveform_amplitude_input.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
        self.window.waveform_amplitude_input.setProperty("waveformField", True)
        self.window.waveform_amplitude_input.setProperty("waveformStepper", True)
        self.window.waveform_amplitude_input.setToolTip("Scale the waveform vertically without changing the raw measurement values")

        slot_widths["window"] = self._waveform_spinbox_slot_width(self.window.display_time_window_input, slot_widths["window"])
        slot_widths["amplitude"] = self._waveform_spinbox_slot_width(self.window.waveform_amplitude_input, slot_widths["amplitude"])

        self.window.go_to_time_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("go"),
            text="Go",
            tooltip="Jump to the selected time",
            icon_only=True,
        )
        self.window.zoom_out_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("zoom_out"),
            text="Out",
            tooltip="Show a longer waveform window",
            icon_only=True,
        )
        self.window.zoom_in_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("zoom_in"),
            text="In",
            tooltip="Show a shorter waveform window",
            icon_only=True,
        )
        self.window.cursor_tool_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("cursor"),
            text="Cursor",
            tooltip="Show a live crosshair cursor over the waveform. Click again, press Esc, or use Reset View to exit.",
            checkable=True,
            icon_only=False,
        )
        self.window.measure_tool_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("measure"),
            text="Measure",
            tooltip="Click two waveform points to measure interval and amplitude difference. Click again, press Esc, or use Reset View to exit.",
            checkable=True,
            icon_only=False,
        )
        self.window.measurement_status_badge = QLabel("")
        self.window.measurement_status_badge.setProperty("waveformBadge", True)
        self.window.measurement_status_badge.setAlignment(Qt.AlignCenter)
        self.window.measurement_status_badge.setVisible(False)
        self.window.hotspot_tool_button = self._make_waveform_tool_button(
            icon=get_waveform_toolbar_icon("hotspot"),
            text="Hotspot",
            tooltip="Focus the most active channel set in the active or accepted run. Click again, All, or Reset View to return.",
            checkable=True,
            icon_only=False,
        )
        self.window.graph_window_preset_buttons = []
        for seconds in (2, 5, 10, 20):
            button = self._make_waveform_preset_button(
                text=f"{seconds} s",
                tooltip=f"Set the visible waveform window to {seconds} seconds",
            )
            button.setProperty("windowPreset", float(seconds))
            button.setProperty("preserveButtonHeight", True)
            button.setFixedHeight(26)
            button.setFixedWidth(42)
            button.hide()
            self.window.graph_window_preset_buttons.append(button)
        self.window.graph_channel_preset_buttons = []
        for preset_value, label in ((8, "8 ch"), (16, "16 ch"), (32, "32 ch"), ("all", "Max")):
            button = self._make_waveform_preset_button(
                text=label,
                tooltip=(
                    f"Show {label.lower()} in the waveform view"
                    if preset_value != "all"
                    else "Show the full current channel subset"
                ),
            )
            button.setProperty("channelPreset", preset_value)
            button.setProperty("preserveButtonHeight", True)
            button.setFixedHeight(26)
            button.setFixedWidth(42)
            self.window.graph_channel_preset_buttons.append(button)

        self.window.display_time_window_slot = self._mount_waveform_toolbar_slot(
            self.window.waveform_toolbar_frame,
            self.window.display_time_window_input,
            width=slot_widths["window"],
            height=toolbar_item_height,
        )
        self.window.time_increment_slot = self._mount_waveform_toolbar_slot(
            self.window.waveform_toolbar_frame,
            self.window.Time_Increment_Input,
            width=slot_widths["step"],
            height=toolbar_item_height,
        )
        self.window.waveform_amplitude_slot = self._mount_waveform_toolbar_slot(
            self.window.waveform_toolbar_frame,
            self.window.waveform_amplitude_input,
            width=slot_widths["amplitude"],
            height=toolbar_item_height,
        )
        self.window.go_to_time_slot = self._mount_waveform_toolbar_slot(
            self.window.waveform_toolbar_frame,
            self.window.go_to_time_input,
            width=slot_widths["jump"],
            height=toolbar_item_height,
        )
        self.window.visible_channel_slot = self._mount_waveform_toolbar_slot(
            self.window.waveform_toolbar_frame,
            self.window.n_channel_input,
            width=slot_widths["visible_channels"],
            height=toolbar_item_height,
        )
        self.window.montage_status_slot = self._mount_waveform_toolbar_slot(
            self.window.waveform_toolbar_frame,
            self.window.montage_status_badge,
            width=slot_widths["montage_status"],
            height=toolbar_item_height,
        )
        self.window.montage_status_slot.setVisible(False)
        self.window.montage_break_slot = self._mount_waveform_toolbar_slot(
            self.window.waveform_toolbar_frame,
            self.window.montage_break_badge,
            width=slot_widths["montage_break"],
            height=toolbar_item_height,
        )
        self.window.montage_break_slot.setVisible(False)
        self.window.event_position_slot = self._mount_waveform_toolbar_slot(
            self.window.waveform_toolbar_frame,
            self.window.event_position_label,
            width=slot_widths["event_position"],
            height=toolbar_item_height,
        )
        self.window.measurement_status_slot = self._mount_waveform_toolbar_slot(
            self.window.waveform_toolbar_frame,
            self.window.measurement_status_badge,
            width=slot_widths["measurement"],
            height=toolbar_item_height,
        )
        self.window.measurement_status_slot.setVisible(False)

        waveform_row, waveform_layout = self._make_waveform_toolbar_group(
            self.window.waveform_toolbar_frame,
            spacing=3,
        )
        channel_row, channel_layout = self._make_waveform_toolbar_group(
            self.window.waveform_toolbar_frame,
            spacing=3,
        )
        event_row, event_layout = self._make_waveform_toolbar_group(
            self.window.waveform_toolbar_frame,
            spacing=3,
        )

        for button in (
            self.window.raw_tool_button,
            self.window.filtered_tool_button,
            self.window.filter60_tool_button,
            self.window.normalize_tool_button,
        ):
            waveform_layout.addWidget(button)
        waveform_layout.addSpacing(4)
        waveform_layout.addWidget(self.window.display_time_window_slot)
        waveform_layout.addWidget(self.window.time_increment_slot)
        waveform_layout.addWidget(self.window.waveform_amplitude_slot)
        waveform_layout.addSpacing(4)
        waveform_layout.addWidget(self.window.go_to_time_slot)
        waveform_layout.addWidget(self.window.go_to_time_button)
        waveform_layout.addWidget(self.window.zoom_out_button)
        waveform_layout.addWidget(self.window.zoom_in_button)
        waveform_layout.addStretch(1)

        channel_layout.addWidget(self.window.review_channels_button)
        channel_layout.addWidget(self.window.montage_tool_button)
        channel_layout.addWidget(self.window.referential_tool_button)
        channel_layout.addWidget(self.window.average_reference_button)
        channel_layout.addWidget(self.window.auto_bipolar_button)
        channel_layout.addWidget(self.window.montage_status_slot)
        channel_layout.addWidget(self.window.montage_break_slot)
        channel_layout.addSpacing(4)
        channel_layout.addWidget(self.window.highlight_channel_button)
        channel_layout.addWidget(self.window.neighbor_channels_button)
        channel_layout.addWidget(self.window.clean_view_button)
        channel_layout.addWidget(self.window.all_channels_button)
        channel_layout.addSpacing(4)
        for button in self.window.graph_channel_preset_buttons:
            channel_layout.addWidget(button)
        channel_layout.addSpacing(4)
        channel_layout.addWidget(self.window.visible_channel_slot)
        channel_layout.addStretch(1)

        for button in (
            self.window.prev_event_button,
            self.window.center_event_button,
            self.window.next_event_button,
            self.window.pending_event_button,
        ):
            event_layout.addWidget(button)
        event_layout.addWidget(self.window.event_position_slot)
        event_layout.addSpacing(4)
        event_layout.addWidget(self.window.event_channels_button)
        event_layout.addWidget(self.window.cursor_tool_button)
        event_layout.addSpacing(4)
        event_layout.addWidget(self.window.measure_tool_button)
        event_layout.addWidget(self.window.measurement_status_slot)
        event_layout.addWidget(self.window.hotspot_tool_button)
        event_layout.addWidget(self.window.overlap_review_button)
        event_layout.addWidget(self.window.open_review_button)
        event_layout.addWidget(self.window.snapshot_button)
        event_layout.addStretch(1)

        toolbar_layout.addWidget(waveform_row)
        toolbar_layout.addWidget(channel_row)
        toolbar_layout.addWidget(event_row)

        for button in (
            self.window.prev_event_button,
            self.window.center_event_button,
            self.window.next_event_button,
            self.window.pending_event_button,
            self.window.open_review_button,
            self.window.snapshot_button,
            self.window.raw_tool_button,
            self.window.normalize_tool_button,
            self.window.filtered_tool_button,
            self.window.filter60_tool_button,
            self.window.review_channels_button,
            self.window.referential_tool_button,
            self.window.average_reference_button,
            self.window.auto_bipolar_button,
            self.window.highlight_channel_button,
            self.window.neighbor_channels_button,
            self.window.clean_view_button,
            self.window.overlap_review_button,
            self.window.montage_tool_button,
            self.window.event_channels_button,
            self.window.all_channels_button,
            self.window.cursor_tool_button,
            self.window.measure_tool_button,
            self.window.hotspot_tool_button,
            self.window.go_to_time_button,
            self.window.zoom_out_button,
            self.window.zoom_in_button,
        ):
            button.setProperty("preserveButtonHeight", True)
            button.setFixedHeight(toolbar_item_height)
            button.setFixedWidth(35)

        for widget in (
            self.window.n_channel_input,
            self.window.display_time_window_input,
            self.window.Time_Increment_Input,
            self.window.waveform_amplitude_input,
            self.window.go_to_time_input,
        ):
            widget.setProperty("preserveInputHeight", True)
            widget.setFixedHeight(toolbar_item_height)

        top_controls.addWidget(self.window.waveform_toolbar_frame)
        secondary_controls.addStretch(1)
        waveform_layout.activate()
        channel_layout.activate()
        event_layout.activate()
        toolbar_layout.activate()
        top_controls.activate()
        secondary_controls.activate()
        self.window.waveform_toolbar_frame.updateGeometry()
        QtCore.QTimer.singleShot(0, self._lock_waveform_toolbar_button_widths)

    def _reposition_waveform_placeholder(self):
        placeholder = getattr(self.window, "waveform_placeholder", None)
        anchor_rect = self._get_waveform_placeholder_anchor_rect()
        if placeholder is None or anchor_rect is None:
            return
        max_width = min(560, max(400, anchor_rect.width() - 72))
        placeholder.setMaximumWidth(max_width)
        placeholder.adjustSize()
        available_width = max(0, anchor_rect.width() - placeholder.width())
        available_height = max(0, anchor_rect.height() - placeholder.height())
        x_pos = anchor_rect.x() + (available_width // 2)
        y_pos = anchor_rect.y() + max(24, min(available_height // 4, 56))
        placeholder.move(x_pos, y_pos)

    def eventFilter(self, obj, event):
        target = getattr(self, "_placeholder_target", None)
        if obj is target and event.type() in {QEvent.Resize, QEvent.Show}:
            QtCore.QTimer.singleShot(0, self._reposition_waveform_placeholder)
        window = getattr(self, "window", None)
        toolbar = getattr(window, "waveform_toolbar_frame", None) if window is not None else None
        if obj is toolbar and event.type() in {QEvent.Show, QEvent.Resize, QEvent.LayoutRequest}:
            self._lock_waveform_toolbar_button_widths()
        image_label = getattr(window, "image_label", None) if window is not None else None
        if obj is image_label and event.type() in {QEvent.Show, QEvent.Resize, QEvent.LayoutRequest}:
            QtCore.QTimer.singleShot(0, self._refresh_hub_model_banner)
        return super(MainWindowView, self).eventFilter(obj, event)

    def _build_workflow_header(self):
        header = QFrame(self.window)
        header.setObjectName("workflowHeader")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)
        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        title = QLabel("PyBrain")
        title.setProperty("heroTitle", True)
        subtitle = QLabel("EEG review workspace for HFO, spindle, and spike analysis.")
        subtitle.setWordWrap(True)
        subtitle.setProperty("heroSubtitle", True)
        self.window.workflow_subtitle_label = subtitle
        title_wrap = QVBoxLayout()
        title_wrap.addWidget(title)
        title_wrap.addWidget(subtitle)
        top_row.addLayout(title_wrap, 1)

        biomarker_label = QLabel("Analysis mode")
        biomarker_label.setProperty("fieldLabel", True)
        self.window.combo_box_biomarker.setMinimumWidth(160)
        self._apply_compact_input_height(self.window.combo_box_biomarker)
        selector_wrap = QVBoxLayout()
        selector_wrap.addWidget(biomarker_label)
        selector_wrap.addWidget(self.window.combo_box_biomarker)
        top_row.addLayout(selector_wrap)
        layout.addLayout(top_row)

        self.window.workflow_status_label = QLabel("Open a recording to begin")
        self.window.workflow_status_label.setProperty("workflowStatus", True)
        layout.addWidget(self.window.workflow_status_label)

        chips_layout = QHBoxLayout()
        chips_layout.setSpacing(8)
        chip_defs = [
            ("status_loaded_chip", "Load data"),
            ("status_channels_chip", "Choose channels"),
            ("status_filter_chip", "Raw signal"),
            ("status_detect_chip", "Detect events"),
            ("status_classify_chip", "Classification optional"),
            ("status_annotate_chip", "Review & annotate"),
        ]
        for attr_name, text in chip_defs:
            chip = QLabel(text)
            chip.setProperty("chip", True)
            chip.setProperty("active", False)
            setattr(self.window, attr_name, chip)
            chips_layout.addWidget(chip)
        chips_layout.addStretch(1)
        layout.addLayout(chips_layout)

        workflow = QFrame(self.window)
        workflow.setObjectName("workflowSteps")
        workflow.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        workflow.setMaximumHeight(76)
        workflow_layout = QHBoxLayout(workflow)
        workflow_layout.setContentsMargins(12, 6, 12, 6)
        workflow_layout.setSpacing(8)
        steps = [
            ("1", "Load Data", "Open EDF, BrainVision, or FIF recordings"),
            ("2", "Choose Channels", "Focus on review targets"),
            ("3", "Configure Analysis", "Tune detector and review settings"),
            ("4", "Review Results", "Inspect signals and event overlays"),
            ("5", "Annotate / Export", "Finalize labels and outputs"),
        ]
        for number, title_text, desc in steps:
            card = QFrame()
            card.setProperty("softCard", True)
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(6, 2, 6, 2)
            badge = QLabel(number)
            badge.setFixedSize(22, 22)
            badge.setAlignment(Qt.AlignCenter)
            badge.setProperty("stepBadge", True)
            step_title = QLabel(title_text)
            step_title.setProperty("stepTitle", True)
            step_desc = QLabel(desc)
            step_desc.setWordWrap(True)
            step_desc.setProperty("stepDescription", True)
            card_layout.addWidget(badge, alignment=Qt.AlignLeft)
            card_layout.addWidget(step_title)
            card_layout.addWidget(step_desc)
            workflow_layout.addWidget(card, 1)

        self.window.workflow_header_frame = header
        self.window.workflow_steps_frame = workflow
        self.window.gridLayout_8.addWidget(header, 1, 0)
        self.window.gridLayout_8.addWidget(workflow, 2, 0)

    def _build_decision_overview(self):
        overview = QFrame(self.window)
        overview.setProperty("metricCard", True)
        overview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        overview.setMaximumHeight(120)
        overview_layout = QHBoxLayout(overview)
        overview_layout.setContentsMargins(14, 10, 14, 10)
        overview_layout.setSpacing(10)

        cards = [
            ("decision_runs_value", "Runs in Session", "--"),
            ("decision_active_value", "Active Detector", "--"),
            ("decision_accepted_value", "Accepted Detector", "--"),
            ("decision_channel_value", "Top Channel", "--"),
        ]
        for attr_name, label_text, default in cards:
            card = QFrame()
            card.setProperty("metricCard", True)
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            card.setMinimumHeight(82)
            card.setMaximumHeight(92)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 10, 12, 10)
            title = QLabel(label_text)
            title.setProperty("metricLabel", True)
            value = QLabel(default)
            value.setProperty("metricValue", True)
            setattr(self.window, attr_name, value)
            card_layout.addWidget(title)
            card_layout.addWidget(value)
            overview_layout.addWidget(card, 1)

        self.window.decision_overview_frame = overview
        self.window.gridLayout_8.addWidget(overview, 3, 0)

    def _build_empty_state_card(self):
        card = QFrame(self.window)
        card.setProperty("metricCard", True)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        card.setMaximumWidth(1080)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(12)

        eyebrow = QLabel("Start a case")
        eyebrow.setProperty("pageEyebrow", True)
        layout.addWidget(eyebrow)

        title = QLabel("Open a recording to start review")
        title.setProperty("pageTitle", True)
        self.window.empty_state_title = title
        layout.addWidget(title)

        subtitle = QLabel("Load EDF, BrainVision, or FIF recordings, then create analysis runs and review channels, detections, and exported summaries from one workspace.")
        subtitle.setWordWrap(True)
        subtitle.setProperty("pageSubtitle", True)
        self.window.empty_state_subtitle = subtitle
        layout.addWidget(subtitle)

        quick_actions = QHBoxLayout()
        quick_actions.setSpacing(10)
        self.window.empty_open_button = QPushButton("Open Recording")
        self.window.empty_load_session_button = QPushButton("Load Session")
        self.window.empty_quick_button = QPushButton("Quick Detection")
        self.window.empty_open_button.setProperty("inspectorPrimary", True)
        quick_actions.addWidget(self.window.empty_open_button)
        quick_actions.addWidget(self.window.empty_load_session_button)
        quick_actions.addWidget(self.window.empty_quick_button)
        quick_actions.addStretch(1)
        layout.addLayout(quick_actions)

        next_row = QHBoxLayout()
        next_row.setSpacing(12)
        for attr_name, title_text, desc in [
            ("empty_step_one", "1. Load data", "Start with EDF, BrainVision, or FIF recordings."),
            ("empty_step_two", "2. Review targets", "Choose channels and montage only after signal is visible."),
            ("empty_step_three", "3. Detect and annotate", "Create runs, compare them, then export the accepted result."),
        ]:
            step_card = QFrame()
            step_card.setProperty("softCard", True)
            step_layout = QVBoxLayout(step_card)
            step_layout.setContentsMargins(14, 12, 14, 12)
            step_title = QLabel(title_text)
            step_title.setProperty("sectionTitle", True)
            step_desc = QLabel(desc)
            step_desc.setWordWrap(True)
            step_desc.setProperty("helperText", True)
            step_layout.addWidget(step_title)
            step_layout.addWidget(step_desc)
            setattr(self.window, attr_name, step_card)
            next_row.addWidget(step_card, 1)
        layout.addLayout(next_row)
        layout.addStretch(1)

        self.window.empty_state_card = card

    def _restructure_main_layout(self):
        layout = self.window.gridLayout_2
        waveform_item = layout.itemAtPosition(0, 0)
        tab_item = layout.itemAtPosition(0, 1)
        if waveform_item is None or tab_item is None:
            return
        waveform_widget = waveform_item.widget()
        tab_widget = tab_item.widget()
        layout.removeWidget(waveform_widget)
        layout.removeWidget(tab_widget)
        self.window.workspace_tabs = tab_widget
        if hasattr(tab_widget, "tabBar"):
            tab_widget.tabBar().setExpanding(False)
            tab_widget.tabBar().setUsesScrollButtons(True)
            tab_widget.setMaximumHeight(560)

        loaded_page = QWidget()
        loaded_layout = QVBoxLayout(loaded_page)
        loaded_layout.setContentsMargins(0, 0, 0, 0)
        loaded_layout.setSpacing(0)
        loaded_splitter = QtWidgets.QSplitter(Qt.Vertical)
        loaded_splitter.setObjectName("loadedWorkspaceSplitter")
        loaded_splitter.setChildrenCollapsible(False)
        loaded_splitter.setHandleWidth(8)
        loaded_splitter.addWidget(waveform_widget)
        loaded_splitter.addWidget(tab_widget)
        loaded_splitter.setStretchFactor(0, 8)
        loaded_splitter.setStretchFactor(1, 4)
        self.window.loaded_workspace_splitter = loaded_splitter
        loaded_layout.addWidget(loaded_splitter)

        start_page = QWidget()
        start_layout = QVBoxLayout(start_page)
        start_layout.setContentsMargins(0, 8, 0, 0)
        start_layout.addStretch(1)
        centered_row = QHBoxLayout()
        centered_row.addStretch(1)
        centered_row.addWidget(self.window.empty_state_card, 0)
        centered_row.addStretch(1)
        start_layout.addLayout(centered_row)
        start_layout.addStretch(2)

        workspace_stack = QStackedWidget()
        workspace_stack.addWidget(start_page)
        workspace_stack.addWidget(loaded_page)
        self.window.workspace_stack = workspace_stack

        layout.addWidget(workspace_stack, 0, 0, 1, 2)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 1)

        self.window.waveformWiget.layout().setStretch(4, 22)
        self.window.waveformWiget.layout().setStretch(5, 2)
        self.window.widget.setMinimumHeight(360)
        self.window.tabWidget.setMinimumHeight(260)

    def _build_status_log_panel(self):
        parent_layout = self.window.gridLayout_6
        terminal = self.window.STDTextEdit
        parent_layout.removeWidget(terminal)
        terminal_group = QGroupBox("2. Status and activity")
        self.window.status_log_group = terminal_group
        terminal_layout = QVBoxLayout(terminal_group)
        terminal_layout.setContentsMargins(10, 10, 10, 10)
        helper = QLabel("Operational messages, warnings, and task progress appear here in chronological order.")
        helper.setProperty("helperText", True)
        terminal_layout.addWidget(helper)
        terminal.setMinimumHeight(90)
        terminal.setMaximumHeight(130)
        terminal.setProperty("consolePanel", True)
        terminal_layout.addWidget(terminal)
        parent_layout.addWidget(terminal_group, 1, 0, 1, 3)

    def _add_run_management_panel(self):
        layout = self.window.gridLayout_27
        run_box = QGroupBox("B. Run review and decisions")
        self.window.run_management_box = run_box
        run_layout = QVBoxLayout(run_box)
        run_layout.setContentsMargins(10, 10, 10, 10)

        helper = QLabel("Switch between stored detector runs, mark the accepted run, and inspect detector agreement.")
        helper.setWordWrap(True)
        helper.setProperty("helperText", True)
        run_layout.addWidget(helper)

        self.window.run_summary_label = QLabel("No detection runs yet.")
        self.window.run_summary_label.setWordWrap(True)
        self.window.run_summary_label.setProperty("helperText", True)
        run_layout.addWidget(self.window.run_summary_label)

        button_row = QHBoxLayout()
        self.window.switch_run_button = QPushButton("Switch Active Run")
        self.window.accept_run_button = QPushButton("Accept Active Run")
        self.window.compare_runs_button = QPushButton("Compare Runs")
        button_row.addWidget(self.window.switch_run_button)
        button_row.addWidget(self.window.accept_run_button)
        button_row.addWidget(self.window.compare_runs_button)
        run_layout.addLayout(button_row)

        self.window.switch_run_button.setEnabled(False)
        self.window.accept_run_button.setEnabled(False)
        self.window.compare_runs_button.setEnabled(False)
        layout.addWidget(run_box, 1, 0, 1, 2)

    def _build_case_console_layout(self):
        layout = self.window.gridLayout_27
        workflow_item = layout.itemAtPosition(0, 0)
        export_item = layout.itemAtPosition(1, 0)
        run_box = getattr(self.window, "run_management_box", None)
        if workflow_item is None or export_item is None or run_box is None:
            return

        workflow_panel = workflow_item.layout() or workflow_item.widget()
        export_panel = export_item.widget() or export_item.layout()
        layout.removeItem(workflow_item)
        layout.removeItem(export_item)
        layout.removeWidget(run_box)

        analysis_panel = QFrame()
        analysis_panel.setProperty("metricCard", True)
        self.window.case_console_panel = analysis_panel
        analysis_layout = QVBoxLayout(analysis_panel)
        analysis_layout.setContentsMargins(14, 12, 14, 12)
        analysis_layout.setSpacing(10)
        analysis_title = QLabel("Case Console")
        analysis_title.setProperty("panelTitle", True)
        analysis_helper = QLabel("Prepare the signal, tune detectors, and configure model passes before reviewing candidate events.")
        analysis_helper.setWordWrap(True)
        analysis_helper.setProperty("helperText", True)
        analysis_layout.addWidget(analysis_title)
        analysis_layout.addWidget(analysis_helper)
        analysis_workspace = self._build_analysis_workspace(workflow_panel)
        analysis_layout.addWidget(analysis_workspace)
        self.window.case_workspace_body = analysis_workspace

        decision_panel = QFrame()
        decision_panel.setProperty("metricCard", True)
        self.window.decision_desk_panel = decision_panel
        decision_layout = QVBoxLayout(decision_panel)
        decision_layout.setContentsMargins(14, 12, 14, 12)
        decision_layout.setSpacing(10)
        decision_title = QLabel("Decision Desk")
        decision_title.setProperty("panelTitle", True)
        decision_helper = QLabel("Track detector runs, choose the accepted analysis, and prepare outputs for clinical review.")
        decision_helper.setWordWrap(True)
        decision_helper.setProperty("helperText", True)
        decision_layout.addWidget(decision_title)
        decision_layout.addWidget(decision_helper)
        decision_layout.addWidget(run_box)
        decision_layout.addWidget(self._build_decision_tables(), 1)
        if isinstance(export_panel, QLayout):
            export_container = QFrame()
            export_container.setProperty("metricCard", True)
            export_container_layout = QVBoxLayout(export_container)
            export_container_layout.setContentsMargins(0, 0, 0, 0)
            export_container_layout.addLayout(export_panel)
            decision_layout.addWidget(export_container)
        else:
            decision_layout.addWidget(export_panel)

        layout.addWidget(analysis_panel, 0, 0)
        layout.addWidget(decision_panel, 0, 1)
        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(1, 2)
        layout.setRowStretch(0, 1)

        self.window.statistics_box.setMinimumWidth(340)
        self.window.statistics_box.setMaximumWidth(520)
        self.window.classifier_groupbox_4.setTitle("Classifier stack")

    def _build_analysis_workspace(self, workflow_panel):
        workspace = QFrame()
        workspace.setProperty("metricCard", True)
        layout = QHBoxLayout(workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        configure_column = QFrame()
        configure_column.setProperty("metricCard", True)
        configure_layout = QVBoxLayout(configure_column)
        configure_layout.setContentsMargins(12, 12, 12, 12)
        configure_layout.setSpacing(10)
        configure_title = QLabel("Configure")
        configure_title.setProperty("sectionTitle", True)
        configure_layout.addWidget(configure_title)
        if isinstance(workflow_panel, QLayout):
            configure_layout.addLayout(workflow_panel)
        else:
            configure_layout.addWidget(workflow_panel)

        review_column = QFrame()
        review_column.setProperty("metricCard", True)
        review_layout = QVBoxLayout(review_column)
        review_layout.setContentsMargins(12, 12, 12, 12)
        review_layout.setSpacing(10)
        review_title = QLabel("Review settings")
        review_title.setProperty("sectionTitle", True)
        review_hint = QLabel("Detector-specific details stay in the editor tabs. Use this column for summary, classifier setup, and export readiness.")
        review_hint.setWordWrap(True)
        review_hint.setProperty("helperText", True)
        review_layout.addWidget(review_title)
        review_layout.addWidget(review_hint)
        review_layout.addWidget(self.window.classifier_groupbox_4)
        review_layout.addWidget(self.window.statistics_box)

        layout.addWidget(configure_column, 3)
        layout.addWidget(review_column, 2)
        return workspace

    def _build_decision_tables(self):
        panel = QFrame()
        panel.setProperty("metricCard", True)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        run_title = QLabel("Run registry")
        run_title.setProperty("sectionTitle", True)
        layout.addWidget(run_title)

        self.window.run_table = QTableWidget(0, 4)
        self.window.run_table.setHorizontalHeaderLabels(["Detector", "Events", "Channels", "Status"])
        self.window.run_table.verticalHeader().setVisible(False)
        self.window.run_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.window.run_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.window.run_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.window.run_table.setAlternatingRowColors(True)
        self.window.run_table.setMinimumHeight(150)
        self.window.run_table.setProperty("summaryTable", True)
        self.window.run_table.horizontalHeader().setStretchLastSection(True)
        self.window.run_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.window.run_table)

        channel_title = QLabel("Channel ranking")
        channel_title.setProperty("sectionTitle", True)
        layout.addWidget(channel_title)

        self.window.channel_table = QTableWidget(0, 4)
        self.window.channel_table.setHorizontalHeaderLabels(["Channel", "Accepted", "Total", "Priority"])
        self.window.channel_table.verticalHeader().setVisible(False)
        self.window.channel_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.window.channel_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.window.channel_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.window.channel_table.setAlternatingRowColors(True)
        self.window.channel_table.setMinimumHeight(150)
        self.window.channel_table.setProperty("summaryTable", True)
        self.window.channel_table.horizontalHeader().setStretchLastSection(True)
        self.window.channel_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.window.channel_table)

        compare_title = QLabel("Run overlap")
        compare_title.setProperty("sectionTitle", True)
        layout.addWidget(compare_title)

        self.window.comparison_table = QTableWidget(0, 5)
        self.window.comparison_table.setHorizontalHeaderLabels(["Left", "Right", "Overlap", "Only", "Jaccard"])
        self.window.comparison_table.verticalHeader().setVisible(False)
        self.window.comparison_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.window.comparison_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.window.comparison_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.window.comparison_table.setAlternatingRowColors(True)
        self.window.comparison_table.setMinimumHeight(130)
        self.window.comparison_table.setProperty("summaryTable", True)
        self.window.comparison_table.horizontalHeader().setStretchLastSection(True)
        self.window.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.window.comparison_table)

        return panel

    def _retitle_workflow_sections(self):
        self.window.tabWidget.setTabText(0, "Prepare")
        self.window.tabWidget.setTabText(1, "Detect")
        self.window.tabWidget.setTabText(2, "Classify")
        self.window.main_edf_info.setTitle("Recording")
        self.window.filters_groupbox.setTitle("Filter and Signal")
        self.window.statistics_box.setTitle("Summary and Export")
        self.window.groupBox.setTitle("Local Model Files")
        self.window.groupBox_2.setTitle("Model Cards")
        self.window.Choose_Channels_Button.setText("Choose Channels")
        self.window.waveform_plot_button.setText("Update Plot")
        self.window.annotation_button.setText("Open Review")
        self.window.save_csv_button.setText("Export Workbook")
        if hasattr(self.window, "save_report_button"):
            self.window.save_report_button.setText("Export Report")
        self.window.save_npz_button.setText("Save Session")
        self.window.overview_filter_button.setText("Filter")
        self.window.detect_all_button.setText("Run Classification")
        self.window.default_cpu_button.setText("Hugging Face CPU")
        self.window.default_gpu_button.setText("Hugging Face GPU")
        self.window.classifier_save_button.setText("Apply")
        self.window.toggle_filtered_checkbox.setText("Show filtered")
        self.window.Filter60Button.setText("60 Hz")
        self.window.normalize_vertical_input.setText("Normalize")
        self.window.bipolar_button.setText("Bipolar")
        self.window.Choose_Channels_Button.setText("Channels")
        self.window.waveform_plot_button.setText("Refresh")
        self.window.label_21.setText("Workers")
        self.window.label_10.setText("Visible")
        self.window.label_9.setText("Window")
        self.window.label_14.setText("Advance")
        self.window.label_22.setText("")
        self.window.main_filename.setText("No recording loaded")
        self.window.main_sampfreq.setText("--")
        self.window.main_numchannels.setText("--")
        self.window.main_length.setText("--")
        self.window.statistics_label.setText("No detections yet")
        self.window.toolBar_biomarker.hide()
        self.window.biomarker_selection_widget.hide()
        self.window.tabWidget.setDocumentMode(True)
        self.window.tabWidget.setCurrentIndex(0)
        self.window.tabWidget.setUsesScrollButtons(False)
        self.window.tabWidget.tabBar().setExpanding(True)
        self.window.detector_subtabs.setDocumentMode(True)
        self.window.detector_subtabs.tabBar().setExpanding(True)

    def set_workspace_state(self, has_recording, biomarker_type):
        if hasattr(self.window, "workflow_status_label"):
            if has_recording:
                self.window.workflow_status_label.setText("")
                self.window.workflow_status_label.setVisible(False)
            else:
                self.window.workflow_status_label.setText(
                    "Open a recording or load a saved session to start the workspace. Signal prep, detection, and classification controls are ready on the right."
                )
                self.window.workflow_status_label.setVisible(True)

        if hasattr(self.window, "waveform_header_title"):
            self.window.waveform_header_title.setText("EEG/iEEG Review" if has_recording else "EEG/iEEG")

        if hasattr(self.window, "recording_header_label"):
            if has_recording:
                filename = self.window.main_filename.text().strip()
                sfreq = self.window.main_sampfreq.text().strip()
                n_channels = self.window.main_numchannels.text().strip()
                length = self.window.main_length.text().strip()
                summary_parts = []
                if filename:
                    summary_parts.append(filename)
                if sfreq:
                    summary_parts.append(f"{sfreq} Hz")
                if n_channels:
                    summary_parts.append(f"{n_channels} channels")
                if length:
                    summary_parts.append(length)
                self.window.recording_header_label.setText(" • ".join(summary_parts))
                self.window.recording_header_label.setVisible(bool(summary_parts))
            else:
                self.window.recording_header_label.setText("")
                self.window.recording_header_label.setVisible(False)

        if hasattr(self.window, "waveform_mode_frame"):
            self.window.waveform_mode_frame.setVisible(bool(has_recording))

        if hasattr(self.window, "legend_host_frame"):
            self.window.legend_host_frame.setVisible(True)
            self.window.legend_host_frame.setEnabled(has_recording)

        if hasattr(self.window, "waveform_placeholder"):
            self.window.waveform_placeholder.setVisible(not has_recording)
            self.window.waveform_placeholder.raise_()

        if hasattr(self.window, "widget_2"):
            self.window.widget_2.setVisible(True)
            self.window.widget_2.setEnabled(has_recording)
        if hasattr(self.window, "waveform_toolbar_frame"):
            self.window.waveform_toolbar_frame.setVisible(True)
            self.window.waveform_toolbar_frame.setEnabled(has_recording)
        if hasattr(self.window, "waveform_time_scroll_bar"):
            self.window.waveform_time_scroll_bar.setVisible(has_recording)
            self.window.waveform_time_scroll_bar.setEnabled(has_recording)
        if hasattr(self.window, "channel_scroll_bar"):
            self.window.channel_scroll_bar.setVisible(has_recording)
            self.window.channel_scroll_bar.setEnabled(has_recording)
        if hasattr(self.window, "frame_biomarker_type"):
            self.window.frame_biomarker_type.setEnabled(True)
        if hasattr(self.window, "widget"):
            self.window.widget.setEnabled(True)
        if hasattr(self.window, "tabWidget"):
            self.window.tabWidget.setEnabled(True)
        if hasattr(self.window, "main_edf_info"):
            self.window.main_edf_info.setVisible(False)
        if hasattr(self.window, "inspector_dock"):
            self.window.inspector_dock.setVisible(True)
            self.window.inspector_dock.resize(296, self.window.inspector_dock.height())

        if hasattr(self.window, "waveform_placeholder"):
            QtCore.QTimer.singleShot(0, self._reposition_waveform_placeholder)

    def get_biomarker_type(self):
        return self.window.combo_box_biomarker.currentText()

    def create_stacked_widget_detection_param(self, biomarker_type='HFO'):
        if biomarker_type == 'HFO':
            clear_stacked_widget(self.window.stacked_widget_detection_param)
            page_ste = self.create_detection_parameter_page_ste('Detection Parameters (STE)')
            page_mni = self.create_detection_parameter_page_mni('Detection Parameters (MNI)')
            page_hil = self.create_detection_parameter_page_hil('Detection Parameters (HIL)')
            self.window.stacked_widget_detection_param.addWidget(page_ste)
            self.window.stacked_widget_detection_param.addWidget(page_mni)
            self.window.stacked_widget_detection_param.addWidget(page_hil)

            self.window.detector_subtabs.clear()
            tab_ste = self.create_detection_parameter_tab_ste()
            tab_mni = self.create_detection_parameter_tab_mni()
            tab_hil = self.create_detection_parameter_tab_hil()
            self.window.detector_subtabs.addTab(tab_ste, 'STE')
            self.window.detector_subtabs.addTab(tab_mni, 'MNI')
            self.window.detector_subtabs.addTab(tab_hil, 'HIL')

        elif biomarker_type == 'Spindle':
            clear_stacked_widget(self.window.stacked_widget_detection_param)
            page_yasa = self.create_detection_parameter_page_yasa('Detection Parameters (YASA)')
            self.window.stacked_widget_detection_param.addWidget(page_yasa)

            self.window.detector_subtabs.clear()
            tab_yasa = self.create_detection_parameter_tab_yasa()
            self.window.detector_subtabs.addTab(tab_yasa, 'YASA')
        elif biomarker_type == 'Spike':
            clear_stacked_widget(self.window.stacked_widget_detection_param)
            page = QWidget()
            layout = QVBoxLayout(page)
            label = QLabel("Manual review mode.")
            label.setWordWrap(True)
            label.setProperty("helperText", True)
            label.setContentsMargins(8, 10, 8, 10)
            layout.addWidget(label)
            self.window.stacked_widget_detection_param.addWidget(page)

            self.window.detector_subtabs.clear()
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            tab_text = QLabel("Detector controls are not available yet.")
            tab_text.setWordWrap(True)
            tab_text.setProperty("helperText", True)
            tab_text.setContentsMargins(8, 10, 8, 10)
            tab_layout.addWidget(tab_text)
            self.window.detector_subtabs.addTab(tab, 'Spike')
        # Detector tabs are rebuilt when the active biomarker changes, so any
        # newly created inputs/buttons need the compact density treatment again.
        apply_compact_button_heights(self.window, self.ui_density)
        apply_compact_input_heights(self.window, self.ui_density)
        self.update_setup_mode_controls(biomarker_type)
        self._apply_section_visibility("DETECTION")
        self._apply_section_visibility("CLASSIFICATION")
        self._install_button_click_feedback(getattr(self.window, "inspector_container", None))

    def update_setup_mode_controls(self, biomarker_type):
        if hasattr(self.window, "detector_mode_combo"):
            detector_options = {
                "HFO": [("STE", "Short-term energy detector"), ("MNI", "Montreal detector"), ("HIL", "Hilbert detector")],
                "Spindle": [("YASA", "YASA spindle detector")],
                "Spike": [("Review", "Manual review and import workflow")],
            }
            blocker = QSignalBlocker(self.window.detector_mode_combo)
            self.window.detector_mode_combo.clear()
            for label, _hint in detector_options.get(biomarker_type, []):
                self.window.detector_mode_combo.addItem(label)
            self.window.detector_mode_combo.setEnabled(self.window.detector_mode_combo.count() > 1)
            del blocker
            if hasattr(self.window, "detector_mode_hint"):
                hints = {
                    "HFO": "Choose a detector, tune the quick parameters below, then run a new HFO analysis.",
                    "Spindle": "Tune the YASA spindle settings below, then run a new spindle analysis.",
                    "Spike": "Spike mode currently supports review and import, not automated detection.",
                }
                self.window.detector_mode_hint.setText(hints.get(biomarker_type, "Choose the detector for the next run."))

        if hasattr(self.window, "classifier_mode_combo"):
            classifier_options = ["Hugging Face CPU", "Hugging Face GPU", "Custom"]
            if biomarker_type == "Spike":
                classifier_options = ["Review only"]
            blocker = QSignalBlocker(self.window.classifier_mode_combo)
            self.window.classifier_mode_combo.clear()
            self.window.classifier_mode_combo.addItems(classifier_options)
            self.window.classifier_mode_combo.setEnabled(self.window.classifier_mode_combo.count() > 1)
            del blocker
            if hasattr(self.window, "classifier_mode_hint"):
                hints = {
                    "HFO": "Hugging Face presets are recommended. Choose CPU or GPU, adjust the review settings, then classify the active run.",
                    "Spindle": "Hugging Face presets are recommended. Choose CPU or GPU, adjust the review settings, then classify the active spindle run.",
                    "Spike": "Spike mode currently focuses on review and import workflows.",
                }
                self.window.classifier_mode_hint.setText(hints.get(biomarker_type, "Choose a classifier preset."))

    def create_frame_biomarker(self, biomarker_type='HFO'):
        existing_layout = self.window.frame_biomarker_type.layout()
        if existing_layout is None:
            self.window.frame_biomarker_layout = QHBoxLayout(self.window.frame_biomarker_type)
        else:
            clear_layout(existing_layout)
            self.window.frame_biomarker_layout = existing_layout
        self.window.frame_biomarker_type.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.window.frame_biomarker_type.setMinimumWidth(0)
        self.window.frame_biomarker_layout.setContentsMargins(8, 5, 8, 5)
        self.window.frame_biomarker_layout.setSpacing(6)
        if biomarker_type == 'HFO':
            self.create_frame_biomarker_hfo()
        elif biomarker_type == 'Spindle':
            self.create_frame_biomarker_spindle()
        elif biomarker_type == 'Spike':
            self.create_frame_biomarker_spike()

    def create_detection_parameter_page_ste(self, groupbox_title):
        page = QWidget()
        layout = QGridLayout()

        detection_groupbox_ste = QGroupBox(groupbox_title)
        ste_parameter_layout = QGridLayout(detection_groupbox_ste)

        clear_layout(ste_parameter_layout)
        # Create widgets
        label1 = QLabel('Epoch (s)')
        label2 = QLabel('Min Window (s)')
        label3 = QLabel('RMS Window (s)')
        label4 = QLabel('Min Gap Time (s)')
        label5 = QLabel('Min Oscillations')
        label6 = QLabel('Peak Threshold')
        label7 = QLabel('RMS Threshold')
        for label in (label1, label2, label3, label4, label5, label6, label7):
            label.setProperty("fieldLabel", True)

        self.window.ste_epoch_display = QLabel()
        self.window.ste_min_window_display = QLabel()
        self.window.ste_rms_window_display = QLabel()
        self.window.ste_min_gap_time_display = QLabel()
        self.window.ste_min_oscillations_display = QLabel()
        self.window.ste_peak_threshold_display = QLabel()
        self.window.ste_rms_threshold_display = QLabel()
        for display in (
            self.window.ste_epoch_display,
            self.window.ste_min_window_display,
            self.window.ste_rms_window_display,
            self.window.ste_min_gap_time_display,
            self.window.ste_min_oscillations_display,
            self.window.ste_peak_threshold_display,
            self.window.ste_rms_threshold_display,
        ):
            style_value_badge(display, alignment=Qt.AlignCenter)
        self.window.ste_detect_button = QPushButton('Run')

        # Add widgets to the grid layout
        ste_parameter_layout.addWidget(label1, 0, 0)  # Row 0, Column 0
        ste_parameter_layout.addWidget(label2, 0, 1)  # Row 0, Column 1
        ste_parameter_layout.addWidget(self.window.ste_epoch_display, 1, 0)  # Row 1, Column 0
        ste_parameter_layout.addWidget(self.window.ste_min_window_display, 1, 1)  # Row 1, Column 1
        ste_parameter_layout.addWidget(label3, 2, 0)
        ste_parameter_layout.addWidget(label4, 2, 1)
        ste_parameter_layout.addWidget(self.window.ste_rms_window_display, 3, 0)
        ste_parameter_layout.addWidget(self.window.ste_min_gap_time_display, 3, 1)
        ste_parameter_layout.addWidget(label5, 4, 0)
        ste_parameter_layout.addWidget(label6, 4, 1)
        ste_parameter_layout.addWidget(self.window.ste_min_oscillations_display, 5, 0)
        ste_parameter_layout.addWidget(self.window.ste_peak_threshold_display, 5, 1)
        ste_parameter_layout.addWidget(label7, 6, 0)
        ste_parameter_layout.addWidget(self.window.ste_rms_threshold_display, 7, 0)
        ste_parameter_layout.addWidget(self.window.ste_detect_button, 7, 1)

        # Set the layout for the page
        layout.addWidget(detection_groupbox_ste)
        page.setLayout(layout)
        return page

    def create_detection_parameter_page_mni(self, groupbox_title):
        page = QWidget()
        layout = QGridLayout()

        detection_groupbox_mni = QGroupBox(groupbox_title)
        mni_parameter_layout = QGridLayout(detection_groupbox_mni)

        clear_layout(mni_parameter_layout)
        # self.detection_groupbox_mni.setTitle("Detection Parameters (MNI)")

        # Create widgets
        label1 = QLabel('Epoch (s)')
        label2 = QLabel('Min Window (s)')
        label3 = QLabel('Epoch CHF (s)')
        label4 = QLabel('Min Gap Time (s)')
        label5 = QLabel('CHF Percentage')
        label6 = QLabel('Threshold Percentile')
        label7 = QLabel('Window (s)')
        label8 = QLabel('Shift')
        label9 = QLabel('Threshold')
        label10 = QLabel('Min Time')
        for label in (label1, label2, label3, label4, label5, label6, label7, label8, label9, label10):
            label.setProperty("fieldLabel", True)

        self.window.mni_epoch_display = QLabel()
        self.window.mni_min_window_display = QLabel()
        self.window.mni_epoch_chf_display = QLabel()
        self.window.mni_min_gap_time_display = QLabel()
        self.window.mni_chf_percentage_display = QLabel()
        self.window.mni_threshold_percentile_display = QLabel()
        self.window.mni_baseline_window_display = QLabel()
        self.window.mni_baseline_shift_display = QLabel()
        self.window.mni_baseline_threshold_display = QLabel()
        self.window.mni_baseline_min_time_display = QLabel()
        for display in (
            self.window.mni_epoch_display,
            self.window.mni_min_window_display,
            self.window.mni_epoch_chf_display,
            self.window.mni_min_gap_time_display,
            self.window.mni_chf_percentage_display,
            self.window.mni_threshold_percentile_display,
            self.window.mni_baseline_window_display,
            self.window.mni_baseline_shift_display,
            self.window.mni_baseline_threshold_display,
            self.window.mni_baseline_min_time_display,
        ):
            style_value_badge(display, alignment=Qt.AlignCenter)
        self.window.mni_detect_button = QPushButton('Run')

        # Add widgets to the grid layout
        mni_parameter_layout.addWidget(label1, 0, 0)  # Row 0, Column 0
        mni_parameter_layout.addWidget(label2, 0, 1)  # Row 0, Column 1
        mni_parameter_layout.addWidget(self.window.mni_epoch_display, 1, 0)  # Row 1, Column 0
        mni_parameter_layout.addWidget(self.window.mni_min_window_display, 1, 1)  # Row 1, Column 1
        mni_parameter_layout.addWidget(label3, 2, 0)
        mni_parameter_layout.addWidget(label4, 2, 1)
        mni_parameter_layout.addWidget(self.window.mni_epoch_chf_display, 3, 0)
        mni_parameter_layout.addWidget(self.window.mni_min_gap_time_display, 3, 1)
        mni_parameter_layout.addWidget(label5, 4, 0)
        mni_parameter_layout.addWidget(label6, 4, 1)
        mni_parameter_layout.addWidget(self.window.mni_chf_percentage_display, 5, 0)
        mni_parameter_layout.addWidget(self.window.mni_threshold_percentile_display, 5, 1)

        group_box = QGroupBox('Baseline')
        baseline_parameter_layout = QVBoxLayout(group_box)
        baseline_parameter_layout.addWidget(label7)
        baseline_parameter_layout.addWidget(self.window.mni_baseline_window_display)
        baseline_parameter_layout.addWidget(label8)
        baseline_parameter_layout.addWidget(self.window.mni_baseline_shift_display)
        baseline_parameter_layout.addWidget(label9)
        baseline_parameter_layout.addWidget(self.window.mni_baseline_threshold_display)
        baseline_parameter_layout.addWidget(label10)
        baseline_parameter_layout.addWidget(self.window.mni_baseline_min_time_display)

        mni_parameter_layout.addWidget(group_box, 0, 2, 6, 1)  # Row 0, Column 2, span 1 row, 6 columns
        mni_parameter_layout.addWidget(self.window.mni_detect_button, 6, 2)

        # Set the layout for the page
        layout.addWidget(detection_groupbox_mni)
        page.setLayout(layout)
        return page

    def create_detection_parameter_page_hil(self, groupbox_title):
        page = QWidget()
        layout = QGridLayout()

        detection_groupbox_hil = QGroupBox(groupbox_title)
        hil_parameter_layout = QGridLayout(detection_groupbox_hil)

        clear_layout(hil_parameter_layout)
        # self.detection_groupbox_hil.setTitle("Detection Parameters (HIL)")

        # Create widgets
        label1 = QLabel('Epoch Length (s)')
        label2 = QLabel('Min Window (s)')
        label3 = QLabel('Pass Band (Hz)')
        label4 = QLabel('Stop Band (Hz)')
        label5 = QLabel('Sample Frequency')
        label6 = QLabel('SD Threshold')
        for label in (label1, label2, label3, label4, label5, label6):
            label.setProperty("fieldLabel", True)

        self.window.hil_epoch_time_display = QLabel()
        self.window.hil_min_window_display = QLabel()
        self.window.hil_pass_band_display = QLabel()
        self.window.hil_stop_band_display = QLabel()
        self.window.hil_sample_freq_display = QLabel()
        self.window.hil_sd_threshold_display = QLabel()
        for display in (
            self.window.hil_epoch_time_display,
            self.window.hil_min_window_display,
            self.window.hil_pass_band_display,
            self.window.hil_stop_band_display,
            self.window.hil_sample_freq_display,
            self.window.hil_sd_threshold_display,
        ):
            style_value_badge(display, alignment=Qt.AlignCenter)
        self.window.hil_detect_button = QPushButton('Run')

        # Add widgets to the grid layout
        hil_parameter_layout.addWidget(label1, 0, 0)  # Row 0, Column 0
        hil_parameter_layout.addWidget(label2, 0, 1)  # Row 0, Column 1
        hil_parameter_layout.addWidget(self.window.hil_epoch_time_display, 1, 0)  # Row 1, Column 0
        hil_parameter_layout.addWidget(self.window.hil_min_window_display, 1, 1)  # Row 1, Column 1
        hil_parameter_layout.addWidget(label3, 2, 0)
        hil_parameter_layout.addWidget(label4, 2, 1)
        hil_parameter_layout.addWidget(self.window.hil_pass_band_display, 3, 0)
        hil_parameter_layout.addWidget(self.window.hil_stop_band_display, 3, 1)
        hil_parameter_layout.addWidget(label5, 4, 0)
        hil_parameter_layout.addWidget(label6, 4, 1)
        hil_parameter_layout.addWidget(self.window.hil_sample_freq_display, 5, 0)
        hil_parameter_layout.addWidget(self.window.hil_sd_threshold_display, 5, 1)
        hil_parameter_layout.addWidget(self.window.hil_detect_button, 6, 1)

        # Set the layout for the page
        layout.addWidget(detection_groupbox_hil)
        page.setLayout(layout)
        return page

    def create_detection_parameter_page_yasa(self, groupbox_title):
        page = QWidget()
        layout = QGridLayout()

        detection_groupbox_yasa = QGroupBox(groupbox_title)
        yasa_parameter_layout = QGridLayout(detection_groupbox_yasa)

        clear_layout(yasa_parameter_layout)
        # self.detection_groupbox_hil.setTitle("Detection Parameters (HIL)")

        # Create widgets
        label1 = QLabel('Spindle Band (Hz)')
        label2 = QLabel('Broad Band (Hz)')
        label3 = QLabel('Duration (s)')
        label4 = QLabel('Min Distance (ms)')
        label5 = QLabel('Rel Power')
        label6 = QLabel('Correlation')
        label7 = QLabel('RMS')
        for label in (label1, label2, label3, label4, label5, label6, label7):
            label.setProperty("fieldLabel", True)

        self.window.yasa_freq_sp_display = QLabel()
        self.window.yasa_freq_broad_display = QLabel()
        self.window.yasa_duration_display = QLabel()
        self.window.yasa_min_distance_display = QLabel()
        self.window.yasa_thresh_rel_pow_display = QLabel()
        self.window.yasa_thresh_corr_display = QLabel()
        self.window.yasa_thresh_rms_display = QLabel()
        for display in (
            self.window.yasa_freq_sp_display,
            self.window.yasa_freq_broad_display,
            self.window.yasa_duration_display,
            self.window.yasa_min_distance_display,
            self.window.yasa_thresh_rel_pow_display,
            self.window.yasa_thresh_corr_display,
            self.window.yasa_thresh_rms_display,
        ):
            style_value_badge(display, alignment=Qt.AlignCenter)

        self.window.yasa_detect_button = QPushButton('Run')

        # Add widgets to the grid layout
        yasa_parameter_layout.addWidget(label1, 0, 0)  # Row 0, Column 0
        yasa_parameter_layout.addWidget(label2, 0, 1)  # Row 0, Column 1
        yasa_parameter_layout.addWidget(self.window.yasa_freq_sp_display, 1, 0)  # Row 1, Column 0
        yasa_parameter_layout.addWidget(self.window.yasa_freq_broad_display, 1, 1)  # Row 1, Column 1
        yasa_parameter_layout.addWidget(label3, 2, 0)
        yasa_parameter_layout.addWidget(label4, 2, 1)
        yasa_parameter_layout.addWidget(self.window.yasa_duration_display, 3, 0)
        yasa_parameter_layout.addWidget(self.window.yasa_min_distance_display, 3, 1)

        group_box = QGroupBox('Thresholds')
        thresh_parameter_layout = QVBoxLayout(group_box)
        thresh_parameter_layout.addWidget(label5)
        thresh_parameter_layout.addWidget(self.window.yasa_thresh_rel_pow_display)
        thresh_parameter_layout.addWidget(label6)
        thresh_parameter_layout.addWidget(self.window.yasa_thresh_corr_display)
        thresh_parameter_layout.addWidget(label7)
        thresh_parameter_layout.addWidget(self.window.yasa_thresh_rms_display)

        yasa_parameter_layout.addWidget(group_box, 0, 2, 4, 1)  # Row 0, Column 2, span 1 row, 6 columns
        yasa_parameter_layout.addWidget(self.window.yasa_detect_button, 4, 2)

        # Set the layout for the page
        layout.addWidget(detection_groupbox_yasa)
        page.setLayout(layout)
        return page

    def create_detection_parameter_tab_ste(self):
        self.window.ste_rms_window_input = QLineEdit()
        self.window.ste_min_window_input = QLineEdit()
        self.window.ste_min_gap_input = QLineEdit()
        self.window.ste_epoch_length_input = QLineEdit()
        self.window.ste_min_oscillation_input = QLineEdit()
        self.window.ste_rms_threshold_input = QLineEdit()
        self.window.ste_peak_threshold_input = QLineEdit()
        self.window.STE_save_button = QPushButton('Apply')
        self.window.STE_save_button.setVisible(False)
        return self._create_compact_parameter_tab(
            [
                ("RMS window", self.window.ste_rms_window_input, "sec", "det_ste_rms_window", "essential"),
                ("Min window", self.window.ste_min_window_input, "sec", "det_ste_min_window", "essential"),
                ("Min gap", self.window.ste_min_gap_input, "sec", "det_ste_min_gap", "standard"),
                ("Min oscill.", self.window.ste_min_oscillation_input, "", "det_ste_min_oscillation", "standard"),
                ("Peak thresh.", self.window.ste_peak_threshold_input, "", "det_ste_peak_threshold", "essential"),
                ("RMS thresh.", self.window.ste_rms_threshold_input, "", "det_ste_rms_threshold", "advanced"),
                ("Epoch length", self.window.ste_epoch_length_input, "sec", "det_ste_epoch_length", "advanced"),
            ],
            self.window.STE_save_button,
        )

    def create_detection_parameter_tab_mni(self):
        self.window.mni_epoch_time_input = QLineEdit()
        self.window.mni_epoch_chf_input = QLineEdit()
        self.window.mni_chf_percentage_input = QLineEdit()
        self.window.mni_min_window_input = QLineEdit()
        self.window.mni_min_gap_time_input = QLineEdit()
        self.window.mni_threshold_percentage_input = QLineEdit()
        self.window.mni_baseline_window_input = QLineEdit()
        self.window.mni_baseline_shift_input = QLineEdit()
        self.window.mni_baseline_threshold_input = QLineEdit()
        self.window.mni_baseline_min_time_input = QLineEdit()
        self.window.MNI_save_button = QPushButton('Apply')
        self.window.MNI_save_button.setVisible(False)
        return self._create_compact_parameter_tab(
            [
                ("Epoch time", self.window.mni_epoch_time_input, "sec", "det_mni_epoch_time", "standard"),
                ("Min window", self.window.mni_min_window_input, "sec", "det_mni_min_window", "essential"),
                ("CHF percent", self.window.mni_chf_percentage_input, "%", "det_mni_chf_percent", "essential"),
                ("Threshold", self.window.mni_threshold_percentage_input, "%", "det_mni_threshold", "essential"),
                ("Min gap", self.window.mni_min_gap_time_input, "sec", "det_mni_min_gap", "standard"),
                ("Epoch CHF", self.window.mni_epoch_chf_input, "sec", "det_mni_epoch_chf", "advanced"),
                ("Baseline win.", self.window.mni_baseline_window_input, "sec", "det_mni_baseline_window", "advanced"),
                ("Baseline shift", self.window.mni_baseline_shift_input, "", "det_mni_baseline_shift", "advanced"),
                ("Baseline thres.", self.window.mni_baseline_threshold_input, "", "det_mni_baseline_threshold", "advanced"),
                ("Baseline min", self.window.mni_baseline_min_time_input, "sec", "det_mni_baseline_min", "advanced"),
            ],
            self.window.MNI_save_button,
        )

    def create_detection_parameter_tab_hil(self):
        self.window.hil_sample_freq_input = QLineEdit()
        self.window.hil_pass_band_input = QLineEdit()
        self.window.hil_stop_band_input = QLineEdit()
        self.window.hil_epoch_time_input = QLineEdit()
        self.window.hil_sd_threshold_input = QLineEdit()
        self.window.hil_min_window_input = QLineEdit()
        self.window.HIL_save_button = QPushButton('Apply')
        self.window.HIL_save_button.setVisible(False)
        return self._create_compact_parameter_tab(
            [
                ("Pass band", self.window.hil_pass_band_input, "Hz", "det_hil_pass_band", "standard"),
                ("Stop band", self.window.hil_stop_band_input, "Hz", "det_hil_stop_band", "standard"),
                ("SD thresh.", self.window.hil_sd_threshold_input, "", "det_hil_sd_threshold", "essential"),
                ("Min window", self.window.hil_min_window_input, "sec", "det_hil_min_window", "essential"),
                ("Sample freq.", self.window.hil_sample_freq_input, "Hz", "det_hil_sample_freq", "advanced"),
                ("Epoch length", self.window.hil_epoch_time_input, "sec", "det_hil_epoch_length", "advanced"),
            ],
            self.window.HIL_save_button,
        )

    def create_detection_parameter_tab_yasa(self):
        tab = QWidget()
        tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        groupbox = QGroupBox("")
        groupbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        parameter_layout = QGridLayout(groupbox)
        self._configure_compact_form_grid(parameter_layout)

        self.window.yasa_freq_sp_low_input = QLineEdit()
        self.window.yasa_freq_sp_high_input = QLineEdit()
        self.window.yasa_freq_broad_low_input = QLineEdit()
        self.window.yasa_freq_broad_high_input = QLineEdit()
        self.window.yasa_duration_low_input = QLineEdit()
        self.window.yasa_duration_high_input = QLineEdit()
        self.window.yasa_min_distance_input = QLineEdit()
        self.window.yasa_thresh_rel_pow_input = QLineEdit()
        self.window.yasa_thresh_corr_input = QLineEdit()
        self.window.yasa_thresh_rms_input = QLineEdit()
        self.window.YASA_save_button = QPushButton('Apply')
        self.window.YASA_save_button.setVisible(False)
        for widget in (
            self.window.yasa_freq_sp_low_input,
            self.window.yasa_freq_sp_high_input,
            self.window.yasa_freq_broad_low_input,
            self.window.yasa_freq_broad_high_input,
            self.window.yasa_duration_low_input,
            self.window.yasa_duration_high_input,
            self.window.yasa_min_distance_input,
            self.window.yasa_thresh_rel_pow_input,
            self.window.yasa_thresh_corr_input,
            self.window.yasa_thresh_rms_input,
        ):
            widget.setAlignment(Qt.AlignCenter)

        parameter_layout.addWidget(
            self._create_compact_range_field_block(
                "Spindle band",
                self.window.yasa_freq_sp_low_input,
                self.window.yasa_freq_sp_high_input,
                "Hz",
                option_id="det_yasa_spindle_band",
                density="essential",
            ),
            0, 0, 1, 6,
        )
        parameter_layout.addWidget(
            self._create_compact_range_field_block(
                "Broad band",
                self.window.yasa_freq_broad_low_input,
                self.window.yasa_freq_broad_high_input,
                "Hz",
                option_id="det_yasa_broad_band",
                density="standard",
            ),
            1, 0, 1, 6,
        )
        parameter_layout.addWidget(
            self._create_compact_range_field_block(
                "Duration",
                self.window.yasa_duration_low_input,
                self.window.yasa_duration_high_input,
                "sec",
                option_id="det_yasa_duration",
                density="essential",
            ),
            2, 0, 1, 6,
        )
        self._add_compact_field_to_grid(
            parameter_layout,
            3,
            0,
            self._create_compact_field_block("Min distance", self.window.yasa_min_distance_input, "ms", option_id="det_yasa_min_distance", density="essential"),
        )
        self._add_compact_field_to_grid(
            parameter_layout,
            3,
            1,
            self._create_compact_field_block("Rel power", self.window.yasa_thresh_rel_pow_input, "", option_id="det_yasa_rel_power", density="standard"),
        )
        self._add_compact_field_to_grid(
            parameter_layout,
            4,
            0,
            self._create_compact_field_block("Correlation", self.window.yasa_thresh_corr_input, "", option_id="det_yasa_correlation", density="advanced"),
        )
        self._add_compact_field_to_grid(
            parameter_layout,
            4,
            1,
            self._create_compact_field_block("RMS", self.window.yasa_thresh_rms_input, "", option_id="det_yasa_rms", density="advanced"),
        )

        parameter_layout.addWidget(self.window.YASA_save_button, 5, 3, 1, 3, alignment=Qt.AlignRight)
        layout.addWidget(groupbox)
        return tab

    def _create_compact_parameter_tab(self, fields, save_button):
        tab = QWidget()
        tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        groupbox = QGroupBox("")
        groupbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        parameter_layout = QGridLayout(groupbox)
        self._configure_compact_form_grid(parameter_layout)

        built_fields = []
        for field in fields:
            if len(field) == 3:
                label_text, widget, unit_text = field
                option_id = None
                density = "standard"
            else:
                label_text, widget, unit_text, option_id, density = field
            widget.setAlignment(Qt.AlignCenter)
            built_fields.append(
                self._create_compact_field_block(label_text, widget, unit_text, option_id=option_id, density=density)
            )

        for index, field_widget in enumerate(built_fields):
            self._add_compact_field_to_grid(parameter_layout, index // 2, index % 2, field_widget)

        save_button.setVisible(False)
        parameter_layout.addWidget(save_button, (len(fields) + 1) // 2, 3, 1, 3, alignment=Qt.AlignRight)
        layout.addWidget(groupbox)
        return tab

    def create_frame_biomarker_hfo(self):
        self._populate_biomarker_legend([
            (COLOR_MAP['HFO'], "HFO"),
            (COLOR_MAP['artifact'], "Artifact"),
            (COLOR_MAP['spkHFO'], "spkHFO"),
            (COLOR_MAP['eHFO'], "eHFO"),
        ])

    def create_frame_biomarker_spindle(self):
        self._populate_biomarker_legend([
            (COLOR_MAP['artifact'], "Artifact"),
            (COLOR_MAP['spike'], "spk-Spindle"),
            (COLOR_MAP['Real'], "Spindle"),
        ])

    def create_frame_biomarker_spike(self):
        self._populate_biomarker_legend([
            ("#b2684d", "Artifact"),
            ("#6b8194", "Spike"),
            ("#758c74", "Accepted"),
        ])

    def _populate_biomarker_legend(self, entries):
        for color, label_text in entries:
            entry_frame = QFrame()
            entry_frame.setProperty("legendChip", True)
            entry_frame.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            entry_layout = QHBoxLayout(entry_frame)
            entry_layout.setContentsMargins(7, 3, 8, 3)
            entry_layout.setSpacing(5)

            swatch = QFrame(entry_frame)
            swatch.setProperty("legendSwatch", True)
            swatch.setFixedSize(13, 10)
            swatch.setStyleSheet(f"background: {color};")
            label = QLabel(label_text, entry_frame)
            label.setProperty("legendLabel", True)
            entry_layout.addWidget(swatch, 0, Qt.AlignVCenter)
            entry_layout.addWidget(label, 0, Qt.AlignVCenter)

            self.window.frame_biomarker_layout.addWidget(entry_frame, 0, Qt.AlignLeft)
        self.window.frame_biomarker_layout.addStretch(1)

    def add_widget(self, layout, widget):
        attr = getattr(self.window.window_widget, layout)
        method = getattr(attr, 'addWidget')
        method(widget)
