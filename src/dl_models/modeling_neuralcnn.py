import collections.abc
import math
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from .configuration_neuralcnn import ResnetConfig
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from typing import Dict, List, Optional, Set, Tuple, Union
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class NeuralCNNModel(PreTrainedModel):
    config_class = ResnetConfig

    # def __init__(self, in_channels, outputs, freeze = False, channel_selection = True):
    def __init__(self, config):
        super().__init__(config)
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        self.input_channels = config.input_channels
        self.outputs = config.num_classes
        self.channel_selection = config.channel_selection
        self.hidden_size = config.hidden_size
        self.kernel_size = config.kernel_size
        self.stride = config.stride
        self.padding = config.padding
        self.cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.conv1 = nn.Conv2d(self.input_channels, self.hidden_size, self.kernel_size, self.stride, self.padding,
                                   bias=False)
        self.cnn.fc = nn.Sequential(nn.Linear(512, self.hidden_size // 2))
        for param in self.cnn.fc.parameters():
            param.requires_grad = not config.freeze
        self.bn0 = nn.BatchNorm1d(self.hidden_size // 2)
        self.relu0 = nn.LeakyReLU()
        self.fc = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.bn = nn.BatchNorm1d(self.hidden_size // 2)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.hidden_size // 2, self.hidden_size // 4)
        self.bn1 = nn.BatchNorm1d(self.hidden_size // 4)
        self.relu1 = nn.LeakyReLU()

        self.fc_out = nn.Linear(self.hidden_size // 4, self.outputs)
        self.final_ac = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        batch = self.cnn(x)
        batch = self.bn(self.relu(self.fc(batch)))
        batch = self.bn1(self.relu1(self.fc1(batch)))
        logits = self.final_ac(self.fc_out(batch))

        return logits


class NeuralCNNForImageClassification(PreTrainedModel):
    config_class = ResnetConfig

    # def __init__(self, in_channels, outputs, freeze = False, channel_selection = True):
    def __init__(self, config):
        super().__init__(config)
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        self.input_channels = config.input_channels
        self.outputs = config.num_classes
        self.channel_selection = config.channel_selection
        self.hidden_size = config.hidden_size
        self.kernel_size = config.kernel_size
        self.stride = config.stride
        self.padding = config.padding
        self.cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.conv1 = nn.Conv2d(self.input_channels, self.hidden_size, self.kernel_size, self.stride, self.padding,
                                   bias=False)
        self.cnn.fc = nn.Linear(512, self.hidden_size // 2)
        for param in self.cnn.fc.parameters():
            param.requires_grad = not config.freeze
        self.bn0 = nn.BatchNorm1d(self.hidden_size // 2)
        self.relu0 = nn.LeakyReLU()
        self.fc = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.bn = nn.BatchNorm1d(self.hidden_size // 2)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.hidden_size // 2, self.hidden_size // 4)
        self.bn1 = nn.BatchNorm1d(self.hidden_size // 4)
        self.relu1 = nn.LeakyReLU()

        self.fc_out = nn.Linear(self.hidden_size // 4, self.outputs)
        self.final_ac = nn.Sigmoid()

    def forward(self,
                input_features: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                ):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        if self.input_channels == 1:
            input_features = input_features[:, 0:1, :, :]
        batch = self.cnn(input_features)
        batch = self.bn(self.relu(self.fc(batch)))
        batch = self.bn1(self.relu1(self.fc1(batch)))
        logits = self.fc_out(batch)
        out = self.final_ac(logits)
        # if labels is not None:
        #     loss_fct = BCEWithLogitsLoss()
        #     loss = loss_fct(logits, labels)
        #     return {"loss": loss, "logits": logits}
        # return {"logits": logits}
        return out
