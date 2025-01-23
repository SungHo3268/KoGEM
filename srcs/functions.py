import os
import random
import torch
import numpy as np


def init_random(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)      # one gpu
    torch.cuda.manual_seed_all(seed)    # multi gpu
    # torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"\n* Set a random seed to {seed}\n")

