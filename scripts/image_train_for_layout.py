"""
Train a diffusion model on images.
"""

import argparse

import torch.distributed as dist
from omegaconf import OmegaConf

from layout_diffusion import dist_util, logger
from layout_diffusion.train_util import TrainLoop
from layout_diffusion.util import loopy
from layout_diffusion.layout_diffusion_unet import build_model
from layout_diffusion.resample import build_schedule_sampler
from layout_diffusion.dataset.data_loader import build_loaders
from layout_diffusion.respace import build_diffusion





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--config_file", type=str, default='./configs/LayoutDiffusion-v1.yaml')
    args = parser.parse_args()

    cfg = OmegaConf.create({'cli': args.__dict__})
    cfg = OmegaConf.merge(OmegaConf.load(cfg.cli.config_file), cfg)
    print(OmegaConf.to_yaml(cfg))

    dist_util.setup_dist(local_rank=cfg.cli.local_rank)
    logger.configure(dir=cfg.train.log_dir)
    logger.log('current rank == {}, total_num = {}, \n, {}'.format(dist.get_rank(), dist.get_world_size(), cfg))

    logger.log("creating model...")
    model = build_model(cfg)
    model.to(dist_util.dev())
    print(model)

    logger.log("creating diffusion...")
    diffusion = build_diffusion(cfg)

    logger.log("creating schedule sampler...")
    schedule_sampler = build_schedule_sampler(cfg, diffusion)

    logger.log("creating data loader...")
    train_loader = build_loaders(cfg, mode='train')

    logger.log("training...")
    trainer = TrainLoop(
        model=model,
        diffusion=diffusion,
        schedule_sampler=schedule_sampler,
        data=loopy(train_loader),
        batch_size=cfg.data.parameters.train.batch_size,
        **cfg.train
    )
    trainer.run_loop()


if __name__ == "__main__":
    main()
