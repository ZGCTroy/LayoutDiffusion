"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import json
import os

import imageio
import torch
import torch as th
import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision import utils

from layout_diffusion import dist_util, logger
from layout_diffusion.dataset.data_loader import build_loaders
from layout_diffusion.layout_diffusion_unet import build_model
from layout_diffusion.respace import build_diffusion
from layout_diffusion.util import fix_seed
from layout_diffusion.dataset.util import imagenet_deprocess_batch, get_cropped_image


def torchvision_save_image(imgs, *args, **kwargs):
    utils.save_image(imgs, *args, **kwargs)


def imageio_save_image(img_tensor, path):
    '''
    :param img_tensor: (C, H, W) torch.Tensor
    :param path:
    :param args:
    :param kwargs:
    :return:
    '''

    imageio.imsave(
        uri=path,
        im=img_tensor.cpu().detach().numpy().transpose(1, 2, 0),  # (H, W, C) numpy
        # im=(imagenet_deprocess_batch(img_tensor) * 255.0).clamp(0, 255).to(torch.uint8).cpu().detach().numpy().transpose(1, 2, 0),  # (H, W, C) numpy
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--config_file", type=str, default='./configs/LayoutDiffusion-v1.yaml')

    known_args, unknown_args = parser.parse_known_args()

    known_args = OmegaConf.create(known_args.__dict__)
    cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
    if unknown_args:
        unknown_args = OmegaConf.from_dotlist(unknown_args)
        cfg = OmegaConf.merge(cfg, unknown_args)

    if cfg.sample.fix_seed:
        fix_seed()

    dist_util.setup_dist(local_rank=cfg.local_rank)

    data_loader = build_loaders(cfg, mode='test')

    total_num_samples = len(data_loader.dataset)
    log_dir = os.path.join(cfg.sample.log_root, 'conditional_{}'.format(cfg.sample.timestep_respacing), 'sample{}'.format(total_num_samples), cfg.sample.sample_suffix)
    logger.configure(dir=log_dir)
    logger.log('current rank == {}, total_num = {}, \n, {}'.format(dist.get_rank(), dist.get_world_size(), cfg))
    logger.log(OmegaConf.to_yaml(cfg))

    logger.log("creating model...")
    model = build_model(cfg)
    model.to(dist_util.dev())
    logger.log(model)

    if cfg.sample.pretrained_model_path:
        logger.log("loading model from {}".format(cfg.sample.pretrained_model_path))
        try:
            model.load_state_dict(dist_util.load_state_dict(cfg.sample.pretrained_model_path, map_location="cpu"), strict=True)
            print('successfully load the entire model')
        except:
            print('not successfully load the entire model, try to load part of model')
            model.load_state_dict(dist_util.load_state_dict(cfg.sample.pretrained_model_path, map_location="cpu"), strict=False)

    model.to(dist_util.dev())
    if cfg.sample.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating diffusion...")
    diffusion = build_diffusion(cfg, timestep_respacing=cfg.sample.timestep_respacing)

    def model_fn(x, t, obj_class=None, obj_box=None, obj_mask=None, **kwargs):
        assert obj_class is not None
        assert obj_box is not None

        cond_image, cond_extra_outputs = model(x, t, obj_class=obj_class, obj_box=obj_box, obj_mask=obj_mask)
        cond_mean, cond_variance = th.chunk(cond_image, 2, dim=1)

        obj_class = th.ones_like(obj_class).fill_(model.layout_encoder.num_classes_for_layout_object - 1)
        obj_class[:, 0] = 0

        obj_box = th.zeros_like(obj_box)
        obj_box[:, 0] = th.FloatTensor([0, 0, 1, 1])

        if obj_mask is not None:
            obj_mask = th.zeros_like(obj_mask)
            obj_mask[:, 0] = th.ones(obj_mask.shape[-2:])

        uncond_image, uncond_extra_outputs = model(x, t, obj_class=obj_class, obj_box=obj_box, obj_mask=obj_mask)
        uncond_mean, uncond_variance = th.chunk(uncond_image, 2, dim=1)

        mean = cond_mean + cfg.sample.classifier_free_scale * (cond_mean - uncond_mean)
        return [th.cat([mean, cond_variance], dim=1), cond_extra_outputs]

    for dir_name in ['generated_imgs', 'real_imgs', 'gt_annotations', 'obj_imgs_from_unresized_gt_imgs',
                     'obj_imgs_from_resized_gt_imgs', 'gt_obj_mask']:
        os.makedirs(os.path.join(log_dir, dir_name), exist_ok=True)

    logger.log('sample method = {}'.format(cfg.sample.sample_method))
    logger.log("sampling...")

    for batch_idx, batch in enumerate(data_loader):
        print('rank={}, batch_id={}'.format(dist.get_rank(), batch_idx))
        imgs, cond = batch
        imgs = imgs.to(dist_util.dev())
        model_kwargs = {
            'obj_class': cond['obj_class'].to(dist_util.dev()),
            'obj_box': cond['obj_box'].to(dist_util.dev()),
            'reconstruct_object_image': cfg.model.parameters.reconstruct_object_image
        }
        if 'obj_mask' in cfg.data.parameters.used_condition_types:
            model_kwargs['obj_mask']: cond['obj_mask'].to(dist_util.dev())

        sample_fn = (diffusion.p_sample_loop if cfg.sample.sample_method == 'ddpm' else diffusion.ddim_sample_loop)

        for sample_idx in range(cfg.sample.sample_times):
            out = sample_fn(
                model_fn, (cfg.data.parameters.test.batch_size, 3, cfg.data.parameters.image_size, cfg.data.parameters.image_size),
                clip_denoised=cfg.sample.clip_denoised, model_kwargs=model_kwargs, cond_fn=None, device=dist_util.dev()
            )  # (B, 3, H, W)
            sample = out['sample']

            for img_idx in range(cfg.data.parameters.test.batch_size):
                filename = cond['filename'][img_idx]

                # 1. save generated imgs
                imageio_save_image(
                    img_tensor=sample[img_idx],
                    path=os.path.join(log_dir, "generated_imgs/{}_{}.png".format(filename, sample_idx)),
                )

                if sample_idx == 0:
                    # 2. save real imgs
                    imageio_save_image(
                        img_tensor=imgs[img_idx],
                        path=os.path.join(log_dir, "real_imgs/{}.png".format(filename)),
                    )

                    # 3. save annotations of real imgs
                    with open(os.path.join(log_dir, 'gt_annotations/{}.json'.format(filename)), 'w') as f:
                        gt_annotations = {
                            'obj_class': cond['obj_class'].tolist()[img_idx],
                            'obj_box': cond['obj_box'].tolist()[img_idx],
                            'is_valid_obj': cond['is_valid_obj'].tolist()[img_idx],
                        }
                        f.write(json.dumps(gt_annotations))

                    # 4. save sequence of obj imgs from_unresized_gt_imgs
                    obj_imgs_from_unresized_gt_imgs = cond['obj_image'][img_idx]  # (L, 3, M, M)
                    torchvision_save_image(obj_imgs_from_unresized_gt_imgs, os.path.join(log_dir, "obj_imgs_from_unresized_gt_imgs/{}.png".format(filename)), nrow=cfg.data.parameters.layout_length)

                    # 5. save sequence of obj imgs from_resized_gt_imgs
                    obj_imgs_from_resized_gt_imgs = get_cropped_image(
                        obj_boxes=model_kwargs['obj_box'][img_idx:img_idx + 1],  # (N, L, 4), N=1
                        images=imgs[img_idx:img_idx + 1],  # (N,3,H,W), N=1
                        cropped_size=cfg.data.parameters.mask_size_for_layout_object  # M
                    )  # (N, L, 3, mask_size, mask_size), N=1
                    torchvision_save_image(obj_imgs_from_resized_gt_imgs.squeeze(0), os.path.join(log_dir, "obj_imgs_from_resized_gt_imgs/{}.png".format(filename)),
                                           nrow=cfg.data.parameters.layout_length)

                    # 6. save gt obj mask
                    if 'obj_mask' in cfg.data.parameters.used_condition_types:
                        obj_mask = cond['obj_mask'][img_idx].reshape(-1, 1, cfg.data.parameters.mask_size_for_layout_object, cfg.data.parameters.mask_size_for_layout_object) * 255.0  # (L, 1, M, M)
                        obj_mask = torch.repeat_interleave(obj_mask, repeats=3, dim=1)  # (L, 3 , M, M)
                        torchvision_save_image(obj_mask, os.path.join(log_dir, "gt_obj_mask/{}.png".format(filename)), nrow=cfg.data.parameters.layout_length)

        dist.barrier()
        cur_num_samples = (batch_idx+1) * cfg.data.parameters.test.batch_size * dist.get_world_size()

        logger.log(f"batch_id={batch_idx+1} created {cur_num_samples} / {total_num_samples} samples")
        if cur_num_samples >= total_num_samples:
            break

    logger.log("sampling complete")


if __name__ == "__main__":
    main()
