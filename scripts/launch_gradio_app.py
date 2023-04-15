"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import functools

import torch
import torch as th
from omegaconf import OmegaConf

from layout_diffusion.layout_diffusion_unet import build_model
from layout_diffusion.util import fix_seed
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from layout_diffusion.dataset.data_loader import build_loaders
from scripts.get_gradio_demo import get_demo
from layout_diffusion.dataset.util import image_unnormalize_batch
import numpy as np

object_name_to_idx = {'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14,
        'bench': 15, 'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27, 'umbrella': 28, 'handbag': 31,
        'tie': 32, 'suitcase': 33, 'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37, 'kite': 38, 'baseball bat': 39, 'baseball glove': 40, 'skateboard': 41, 'surfboard': 42,
        'tennis racket': 43, 'bottle': 44, 'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54, 'orange': 55, 'broccoli': 56,
        'carrot': 57, 'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65, 'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73,
        'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86, 'scissors': 87,
        'teddy bear': 88, 'hair drier': 89, 'toothbrush': 90, 'banner': 92, 'blanket': 93, 'branch': 94, 'bridge': 95, 'building-other': 96, 'bush': 97, 'cabinet': 98, 'cage': 99, 'cardboard': 100,
        'carpet': 101, 'ceiling-other': 102, 'ceiling-tile': 103, 'cloth': 104, 'clothes': 105, 'clouds': 106, 'counter': 107, 'cupboard': 108, 'curtain': 109, 'desk-stuff': 110, 'dirt': 111,
        'door-stuff': 112, 'fence': 113, 'floor-marble': 114, 'floor-other': 115, 'floor-stone': 116, 'floor-tile': 117, 'floor-wood': 118, 'flower': 119, 'fog': 120, 'food-other': 121, 'fruit': 122,
        'furniture-other': 123, 'grass': 124, 'gravel': 125, 'ground-other': 126, 'hill': 127, 'house': 128, 'leaves': 129, 'light': 130, 'mat': 131, 'metal': 132, 'mirror-stuff': 133, 'moss': 134,
        'mountain': 135, 'mud': 136, 'napkin': 137, 'net': 138, 'paper': 139, 'pavement': 140, 'pillow': 141, 'plant-other': 142, 'plastic': 143, 'platform': 144, 'playingfield': 145, 'railing': 146,
        'railroad': 147, 'river': 148, 'road': 149, 'rock': 150, 'roof': 151, 'rug': 152, 'salad': 153, 'sand': 154, 'sea': 155, 'shelf': 156, 'sky-other': 157, 'skyscraper': 158, 'snow': 159,
        'solid-other': 160, 'stairs': 161, 'stone': 162, 'straw': 163, 'structural-other': 164, 'table': 165, 'tent': 166, 'textile-other': 167, 'towel': 168, 'tree': 169, 'vegetable': 170,
        'wall-brick': 171, 'wall-concrete': 172, 'wall-other': 173, 'wall-panel': 174, 'wall-stone': 175, 'wall-tile': 176, 'wall-wood': 177, 'water-other': 178, 'waterdrops': 179,
        'window-blind': 180, 'window-other': 181, 'wood': 182, 'other': 183, '__image__': 0, '__null__': 184}


@torch.no_grad()
def layout_to_image_generation(cfg, model_fn, noise_schedule, custom_layout_dict):
    print(custom_layout_dict)

    layout_length = cfg.data.parameters.layout_length

    model_kwargs = {
        'obj_bbox': torch.zeros([1, layout_length, 4]),
        'obj_class': torch.zeros([1, layout_length]).long().fill_(object_name_to_idx['__null__']),
        'is_valid_obj': torch.zeros([1, layout_length])
    }
    model_kwargs['obj_class'][0][0] = object_name_to_idx['__image__']
    model_kwargs['obj_bbox'][0][0] = torch.FloatTensor([0, 0, 1, 1])
    model_kwargs['is_valid_obj'][0][0] = 1.0

    for obj_id in range(1, custom_layout_dict['num_obj']-1):
        obj_bbox = custom_layout_dict['obj_bbox'][obj_id]
        obj_class = custom_layout_dict['obj_class'][obj_id]
        if obj_class == 'pad':
            obj_class = '__null__'

        model_kwargs['obj_bbox'][0][obj_id] = torch.FloatTensor(obj_bbox)
        model_kwargs['obj_class'][0][obj_id] = object_name_to_idx[obj_class]
        model_kwargs['is_valid_obj'][0][obj_id] = 1

    print(model_kwargs)


    wrappered_model_fn = model_wrapper(
        model_fn,
        noise_schedule,
        is_cond_classifier=False,
        total_N=1000,
        model_kwargs=model_kwargs
    )
    for key in model_kwargs.keys():
        model_kwargs[key] = model_kwargs[key].cuda()

    dpm_solver = DPM_Solver(wrappered_model_fn, noise_schedule)

    x_T = th.randn((1, 3, cfg.data.parameters.image_size, cfg.data.parameters.image_size)).cuda()

    sample = dpm_solver.sample(
        x_T,
        steps=int(cfg.sample.timestep_respacing[0]),
        eps=float(cfg.sample.eps),
        adaptive_step_size=cfg.sample.adaptive_step_size,
        fast_version=cfg.sample.fast_version,
        clip_denoised=False,
        rtol=cfg.sample.rtol
    )  # (B, 3, H, W), B=1

    sample = sample.clamp(-1, 1)

    generate_img = np.array(sample[0].cpu().permute(1,2,0) * 127.5 + 127.5, dtype=np.uint8)
    # generate_img = np.transpose(generate_img, (1,0,2))
    print(generate_img.shape)




    print("sampling complete")

    return generate_img


@torch.no_grad()
def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='./configs/COCO-stuff_256x256/LayoutDiffusion-v7_small.yaml')
    parser.add_argument("--share", action='store_true')

    known_args, unknown_args = parser.parse_known_args()

    known_args = OmegaConf.create(known_args.__dict__)
    cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
    if unknown_args:
        unknown_args = OmegaConf.from_dotlist(unknown_args)
        cfg = OmegaConf.merge(cfg, unknown_args)

    print(OmegaConf.to_yaml(cfg))

    print("creating model...")
    model = build_model(cfg)
    model.cuda()
    print(model)

    if cfg.sample.pretrained_model_path:
        print("loading model from {}".format(cfg.sample.pretrained_model_path))
        checkpoint = torch.load(cfg.sample.pretrained_model_path, map_location="cpu")

        try:
            model.load_state_dict(checkpoint, strict=True)
            print('successfully load the entire model')
        except:
            print('not successfully load the entire model, try to load part of model')

            model.load_state_dict(checkpoint, strict=False)

    model.cuda()
    if cfg.sample.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def model_fn(x, t, obj_class=None, obj_bbox=None, obj_mask=None, is_valid_obj=None, **kwargs):
        assert obj_class is not None
        assert obj_bbox is not None

        cond_image, cond_extra_outputs = model(
            x, t,
            obj_class=obj_class, obj_bbox=obj_bbox, obj_mask=obj_mask,
            is_valid_obj=is_valid_obj
        )
        cond_mean, cond_variance = th.chunk(cond_image, 2, dim=1)

        obj_class = th.ones_like(obj_class).fill_(model.layout_encoder.num_classes_for_layout_object - 1)
        obj_class[:, 0] = 0

        obj_bbox = th.zeros_like(obj_bbox)
        obj_bbox[:, 0] = th.FloatTensor([0, 0, 1, 1])

        is_valid_obj = th.zeros_like(obj_class)
        is_valid_obj[:, 0] = 1.0

        if obj_mask is not None:
            obj_mask = th.zeros_like(obj_mask)
            obj_mask[:, 0] = th.ones(obj_mask.shape[-2:])

        uncond_image, uncond_extra_outputs = model(
            x, t,
            obj_class=obj_class, obj_bbox=obj_bbox, obj_mask=obj_mask,
            is_valid_obj=is_valid_obj
        )
        uncond_mean, uncond_variance = th.chunk(uncond_image, 2, dim=1)

        mean = cond_mean + cfg.sample.classifier_free_scale * (cond_mean - uncond_mean)

        if cfg.sample.sample_method in ['ddpm', 'ddim']:
            return [th.cat([mean, cond_variance], dim=1), cond_extra_outputs]
        else:
            return mean

    print("creating diffusion...")

    noise_schedule = NoiseScheduleVP(schedule='linear')

    print('sample method = {}'.format(cfg.sample.sample_method))
    print("sampling...")

    return cfg, model_fn, noise_schedule


if __name__ == "__main__":
    cfg, model_fn, noise_schedule = init()

    demo = get_demo(layout_to_image_generation, cfg, model_fn, noise_schedule)

    demo.launch(share=cfg.share)
