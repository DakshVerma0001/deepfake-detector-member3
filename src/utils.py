import os
import json
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)
