import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward_hook(self, step, epoch, **kwargs):
        pass

    def forward(self, inputs, targets):
        raise NotImplementedError

    def VisualizationHook(self, idx=0):
        return None