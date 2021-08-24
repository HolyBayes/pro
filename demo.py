import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import itertools
import torch.nn.functional as F
from tqdm import trange

class Feature_Encoder(nn.Module):
    def __init__(self, input_dim:int, output_dim:int=20):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x)

class Classifier(nn.Module):
    def __init__(self, input_dim=20):
        super().__init__()
        self.net = nn.Sequential(
            # nn.ReLU()
            nn.Linear(2*input_dim, 1)
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x)