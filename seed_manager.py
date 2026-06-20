# seed_manager.py
import random
import numpy as np
import torch

def set_global_seed(seed=42):
    """固定所有可能的随机源"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False