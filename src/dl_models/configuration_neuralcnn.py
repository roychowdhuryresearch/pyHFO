from transformers import PretrainedConfig
from typing import List

class ResnetConfig(PretrainedConfig):
    model_type = "resnet"
    def __init__(
        self,
        layers: List[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        input_channels: int = 1,
        hidden_size: int = 64,
        kernel_size: int = 7,
        stride: int = 2,
        padding: int = 3,
        freeze: bool = False,
        channel_selection: bool = True,
        **kwargs,
    ):
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.freeze = freeze
        self.channel_selection = channel_selection
        super().__init__(**kwargs)