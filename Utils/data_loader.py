import torch

# fake load function returning X(20x3)
def load():
    return torch.rand(1, 4, 3)