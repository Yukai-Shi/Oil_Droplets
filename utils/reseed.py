import random, os
import numpy as np
import torch
from typing import Optional
from torch.backends import cudnn

def reseed_everything(seed: Optional[int]):
    if seed is None:
        return
    
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 可复现性
    cudnn.deterministic = True
    cudnn.benchmark = False  
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass