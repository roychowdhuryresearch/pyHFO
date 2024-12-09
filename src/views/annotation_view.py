import numpy as np


class AnnotationView:
    def __init__(self, window_widget):
        self.window_widget = window_widget
        # self._init_plot_widget(plot_widget)

    def add_widget(self, layout, widget):
        attr = getattr(self.window_widget, layout)
        method = getattr(attr, 'addWidget')
        method(widget)

