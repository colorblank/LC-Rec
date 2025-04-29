"""随机种子设置模块，用于确保实验的可重复性。"""

import random

import numpy as np
import torch


def set_seed(seed):
    """设置随机种子以确保实验可重复。

    设置Python内置random、numpy和PyTorch的随机种子，
    同时禁用PyTorch的CUDNN后端的非确定性算法。

    Args:
        seed (int): 随机种子值。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False