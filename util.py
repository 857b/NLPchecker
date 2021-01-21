import torch
import numpy

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()
