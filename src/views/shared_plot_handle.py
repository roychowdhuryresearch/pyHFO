class SharedPlotHandle:
    def __init__(self, graphics_widget, plot_item):
        self.graphics_widget = graphics_widget
        self.plot_item = plot_item

    def __getattr__(self, name):
        if hasattr(self.plot_item, name):
            return getattr(self.plot_item, name)
        return getattr(self.graphics_widget, name)

    def getPlotItem(self):
        return self.plot_item

    def setBackground(self, *args, **kwargs):
        return self.graphics_widget.setBackground(*args, **kwargs)

    def setAntialiasing(self, *args, **kwargs):
        return self.graphics_widget.setAntialiasing(*args, **kwargs)

    def width(self):
        try:
            return int(self.plot_item.vb.width())
        except Exception:
            return int(self.graphics_widget.width())
