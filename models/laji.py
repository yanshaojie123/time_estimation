import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class lajimodel(nn.Module):
    def __init__(self):
        super(lajimodel, self).__init__()
        self.pipeline = nn.Sequential(
            nn.Linear(1, 5),
            nn.ReLU(),
            nn.Linear(5,5),
            nn.ReLU(),
            nn.Linear(5,1)
        )
    def forward(self, inputs, args):
        lens = inputs['lens']
        lens = np.asarray(lens.cpu(), dtype="float")
        lens = torch.Tensor(lens).to(args.device).reshape(-1,1)
        output = self.pipeline(lens)
        return output