from src.utils.utils_gui import *


class MainWindowController:
    def __init__(self, view, model):
        self.model = model
        self.view = view

        self.supported_biomarker = {
            'HFO': self.create_hfo_window,
            # TODO future version
            # 'Spindle': self.create_spindle_window,
            # 'Spike': self.create_spike_window,
        }

    def init_biomarker_window(self, biomarker_type):
        # To dynamically create frame for different biomarkers, need first init (optimize later)
        self.view.window.frame_biomarker_layout = QHBoxLayout(self.view.window.frame_biomarker_type)
        self.supported_biomarker[biomarker_type]()

    def init_general_window(self):
        self.view.init_general_window()

        self.model.init_error_terminal_display()
        self.model.init_menu_bar()
        self.model.init_waveform_display()

    def get_biomarker_type(self):
        return self.view.get_biomarker_type()

    def set_biomarker_type(self, bio_type):
        self.model.set_biomarker_type_and_init_backend(bio_type)

    def init_biomarker_type(self):
        default_biomarker = self.get_biomarker_type()
        self.set_biomarker_type(default_biomarker)

        # TODO future version
        # safe_connect_signal_slot(self.view.window.combo_box_biomarker.currentIndexChanged, self.switch_biomarker)

    def switch_biomarker(self):
        selected_biomarker = self.get_biomarker_type()
        self.supported_biomarker[selected_biomarker]()

    def create_hfo_window(self):
        # set biomarker type
        self.set_biomarker_type('HFO')

        # create detection parameters stacked widget
        self.view.create_stacked_widget_detection_param('HFO')

        # create biomarker typ frame widget
        self.view.create_frame_biomarker('HFO')

        # manage flag
        self.view.window.is_data_filtered = False
        self.view.window.quick_detect_open = False

        # create center waveform and mini plot
        self.model.create_center_waveform_and_mini_plot()

        # connect signal & slot
        self.model.connect_signal_and_slot('HFO')

        # init params
        self.model.init_param('HFO')

    def create_spindle_window(self):
        # set biomarker type
        self.set_biomarker_type('Spindle')

        # create detection parameters stacked widget
        self.view.create_stacked_widget_detection_param('Spindle')

        # create biomarker typ frame widget
        self.view.create_frame_biomarker('Spindle')

        # manage flag
        self.view.window.is_data_filtered = False
        self.view.window.quick_detect_open = False

        # create center waveform and mini plot
        self.model.create_center_waveform_and_mini_plot()

        # connect signal & slot
        self.model.connect_signal_and_slot('Spindle')

        # init params
        self.model.init_param('Spindle')

    def create_spike_window(self):
        print('not implemented yet')
