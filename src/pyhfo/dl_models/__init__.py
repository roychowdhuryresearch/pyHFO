"""Optional model families, imported only when requested."""

from importlib import import_module


def __getattr__(name):
    modules = {
        "ResnetConfig": ".configuration_neuralcnn",
        "NeuralCNNModel": ".modeling_neuralcnn",
        "NeuralCNNForImageClassification": ".modeling_neuralcnn",
        "BasedSeverityConfig": ".severity",
        "BasedSeverityModel": ".severity",
    }
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(modules[name], __name__), name)
