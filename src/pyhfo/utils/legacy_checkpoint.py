"""Load trusted PyTorch archives written under the pre-4.0 module names."""

import pickle
import sys

# torch.load accepts a pickle module with Unpickler and load attributes.
load = pickle.load


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in {"src.model", "src.model_pyhfo_process"}:
            module = "pyhfo.model"
        return super().find_class(module, name)


def load_legacy_checkpoint(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=sys.modules[__name__])
