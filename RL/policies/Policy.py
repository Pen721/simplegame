import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

class Policy(nn.Module):
    def __init__(self):
        super().__init__()