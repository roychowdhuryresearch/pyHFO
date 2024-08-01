class ThumbnailController:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.model.data_updated.connect(self.update_view)
        self.update_view()

    def update_view(self):
        data = self.model.get_predicted_data()
        start, end = self.model.get_current_window()
        self.view.update_plot(data, start, end)

    def set_current_window(self, start, end):
        self.model.set_current_window(start, end)