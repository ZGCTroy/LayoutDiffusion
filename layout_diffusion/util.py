import importlib
import torch.distributed as dist
import numpy as np
import torch
import os
import random

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    return getattr(importlib.import_module(module, package=None), cls)


def fix_seed(seed=23333):

    if dist.is_initialized():
        seed = seed + dist.get_rank()

    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU随机种子确定
    torch.cuda.manual_seed(seed)  # GPU随机种子确定
    torch.cuda.manual_seed_all(seed)  # 所有的GPU设置种子

    torch.backends.cudnn.benchmark = False  # 模型卷积层预先优化关闭
    torch.backends.cudnn.deterministic = True  # 确定为默认卷积算法

    random.seed(seed)
    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)


def loopy(data_loader):
    is_distributed=False
    if dist.is_initialized() and dist.get_world_size() > 1:
        is_distributed=True
    epoch = 0
    while True:
        if is_distributed:
            epoch += 1
            data_loader.sampler.set_epoch(epoch)
        for x in iter(data_loader):
            yield x
