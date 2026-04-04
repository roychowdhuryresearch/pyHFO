from PyQt5 import QtCore, QtGui, QtWidgets

from src.utils.utils_gui import apply_subwindow_theme, fit_window_to_screen, set_accent_button


class WaveformShortcutSettingsDialog(QtWidgets.QDialog):
    TRACKPAD_SENSITIVITY_OPTIONS = (
        ("gentle", "Gentle"),
        ("default", "Default"),
        ("fast", "Fast"),
    )

    def __init__(
        self,
        shortcut_specs,
        enabled=True,
        bindings=None,
        validate_callback=None,
        trackpad_sensitivity="default",
        parent=None,
    ):
        super().__init__(parent)
        self.shortcut_specs = list(shortcut_specs or [])
        self.validate_callback = validate_callback
        self._bindings = dict(bindings or {})
        self._editors = {}

        self.setWindowTitle("Waveform Shortcuts")
        fit_window_to_screen(
            self,
            default_width=820,
            default_height=660,
            min_width=700,
            min_height=560,
            width_ratio=0.72,
            height_ratio=0.82,
        )

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        title = QtWidgets.QLabel("Adjust waveform shortcuts", self)
        title.setProperty("dialogTitle", True)
        root.addWidget(title)

        helper = QtWidgets.QLabel(
            "Use plain keys or Shift combinations only, so the waveform tools stay separate from macOS system shortcuts. "
            "Leave a field empty to disable that shortcut.",
            self,
        )
        helper.setProperty("helperText", True)
        helper.setWordWrap(True)
        root.addWidget(helper)

        self.enabled_checkbox = QtWidgets.QCheckBox("Enable waveform shortcuts", self)
        self.enabled_checkbox.setChecked(bool(enabled))
        root.addWidget(self.enabled_checkbox)

        trackpad_group = QtWidgets.QGroupBox("Trackpad", self)
        trackpad_layout = QtWidgets.QFormLayout(trackpad_group)
        trackpad_layout.setContentsMargins(12, 12, 12, 12)
        trackpad_layout.setHorizontalSpacing(10)
        trackpad_layout.setVerticalSpacing(8)

        self.trackpad_sensitivity_combo = QtWidgets.QComboBox(trackpad_group)
        for value, label in self.TRACKPAD_SENSITIVITY_OPTIONS:
            self.trackpad_sensitivity_combo.addItem(label, value)
        current_index = self.trackpad_sensitivity_combo.findData(str(trackpad_sensitivity or "default").strip().lower())
        self.trackpad_sensitivity_combo.setCurrentIndex(current_index if current_index >= 0 else 1)
        self.trackpad_sensitivity_combo.setToolTip("Adjust how aggressively trackpad scrolling and pinch zoom respond")
        trackpad_layout.addRow("Sensitivity", self.trackpad_sensitivity_combo)

        trackpad_helper = QtWidgets.QLabel(
            "Gentle slows two-finger panning and pinch zoom. Fast makes waveform navigation more reactive.",
            trackpad_group,
        )
        trackpad_helper.setProperty("helperText", True)
        trackpad_helper.setWordWrap(True)
        trackpad_layout.addRow("", trackpad_helper)
        root.addWidget(trackpad_group)

        scroll_area = QtWidgets.QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_contents = QtWidgets.QWidget(scroll_area)
        scroll_layout = QtWidgets.QGridLayout(scroll_contents)
        scroll_layout.setContentsMargins(14, 14, 14, 14)
        scroll_layout.setHorizontalSpacing(10)
        scroll_layout.setVerticalSpacing(8)

        headers = ("Action", "Shortcut", "Default", "")
        for column, text in enumerate(headers):
            label = QtWidgets.QLabel(text, scroll_contents)
            label.setProperty("fieldLabel", True)
            scroll_layout.addWidget(label, 0, column)

        for row, spec in enumerate(self.shortcut_specs, start=1):
            action_label = QtWidgets.QLabel(spec["label"], scroll_contents)
            action_label.setWordWrap(True)
            editor = QtWidgets.QKeySequenceEdit(scroll_contents)
            current_value = str(self._bindings.get(spec["id"], spec["default"]) or "").strip()
            editor.setKeySequence(QtGui.QKeySequence(current_value))
            default_label = QtWidgets.QLabel(spec["default"] or "Disabled", scroll_contents)
            default_label.setProperty("helperText", True)
            clear_button = QtWidgets.QPushButton("Clear", scroll_contents)
            clear_button.setMaximumWidth(72)
            clear_button.clicked.connect(lambda _checked=False, current_editor=editor: current_editor.clear())

            scroll_layout.addWidget(action_label, row, 0)
            scroll_layout.addWidget(editor, row, 1)
            scroll_layout.addWidget(default_label, row, 2)
            scroll_layout.addWidget(clear_button, row, 3)
            self._editors[spec["id"]] = editor

        scroll_layout.setColumnStretch(0, 2)
        scroll_layout.setColumnStretch(1, 2)
        scroll_layout.setColumnStretch(2, 1)
        scroll_area.setWidget(scroll_contents)
        root.addWidget(scroll_area, 1)

        self.error_label = QtWidgets.QLabel("", self)
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #9f4136;")
        self.error_label.setVisible(False)
        root.addWidget(self.error_label)

        button_row = QtWidgets.QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(8)

        reset_button = QtWidgets.QPushButton("Reset Defaults", self)
        reset_button.clicked.connect(self.reset_to_defaults)
        reset_button.setDefault(False)
        reset_button.setAutoDefault(False)
        button_row.addWidget(reset_button)
        button_row.addStretch(1)

        cancel_button = QtWidgets.QPushButton("Cancel", self)
        cancel_button.clicked.connect(self.reject)
        cancel_button.setDefault(False)
        cancel_button.setAutoDefault(False)
        button_row.addWidget(cancel_button)

        save_button = QtWidgets.QPushButton("Save", self)
        set_accent_button(save_button)
        save_button.clicked.connect(self._attempt_accept)
        save_button.setDefault(True)
        save_button.setAutoDefault(True)
        button_row.addWidget(save_button)
        root.addLayout(button_row)

        apply_subwindow_theme(self)

    def reset_to_defaults(self):
        self.enabled_checkbox.setChecked(True)
        default_index = self.trackpad_sensitivity_combo.findData("default")
        self.trackpad_sensitivity_combo.setCurrentIndex(default_index if default_index >= 0 else 1)
        for spec in self.shortcut_specs:
            editor = self._editors.get(spec["id"])
            if editor is not None:
                editor.setKeySequence(QtGui.QKeySequence(spec["default"]))
        self.error_label.setVisible(False)
        self.error_label.setText("")

    def shortcuts_enabled(self):
        return bool(self.enabled_checkbox.isChecked())

    def current_bindings(self):
        bindings = {}
        for spec in self.shortcut_specs:
            editor = self._editors.get(spec["id"])
            if editor is None:
                continue
            bindings[spec["id"]] = editor.keySequence().toString(QtGui.QKeySequence.PortableText).strip()
        return bindings

    def current_trackpad_sensitivity(self):
        return str(self.trackpad_sensitivity_combo.currentData() or "default")

    def _attempt_accept(self):
        self.error_label.setVisible(False)
        self.error_label.setText("")
        if self.validate_callback is not None:
            error = self.validate_callback(self.shortcuts_enabled(), self.current_bindings())
            if error:
                self.error_label.setText(str(error))
                self.error_label.setVisible(True)
                return
        self.accept()
