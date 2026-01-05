import torch
import random
import numpy as np

def set_seed(seed=42):
    """Sets seeds for reproducibility."""
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
