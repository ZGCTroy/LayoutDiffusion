
from .coco import build_coco_dsets, coco_collate_fn_for_layout
from .vg import build_vg_dsets, vg_collate_fn_for_layout
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

def build_loaders(cfg, mode='train'):
    assert mode in ['train', 'val', 'test']
    params = cfg.data.parameters

    if cfg.data.type == 'COCO-stuff':
        dataset = build_coco_dsets(cfg, mode=mode)
        collate_fn = coco_collate_fn_for_layout
    elif cfg.data.type == 'VG':
        dataset = build_vg_dsets(cfg, mode=mode)
        collate_fn = vg_collate_fn_for_layout
    else:
        raise NotImplementedError

    loader_kwargs = {
        'batch_size': params[mode].batch_size,
        'num_workers': params.loader_num_workers,
        'shuffle': params[mode].shuffle,
        'collate_fn': collate_fn,
    }
    if mode == 'test' and dist.is_initialized() and dist.get_world_size() > 1:
        test_sampler = DistributedSampler(dataset)
        loader_kwargs['sampler'] = test_sampler

    data_loader = DataLoader(dataset, **loader_kwargs)
    return data_loader