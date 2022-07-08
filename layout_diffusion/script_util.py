import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .layout_diffusion_unet import  UNetModel, EncoderUNetModel
from .layout2im_model import Layout2ImUNet
from omegaconf import OmegaConf

cfg = OmegaConf.load('/workspace/mnt/storage/guangcongzheng/zju_zgc/LayoutDiffusion/configs/LayoutDiffusion-v1.yaml')



def build_model(cfg):
    model = UNetModel(**cfg.model)


#
# def create_model_and_diffusion(
#         image_size,
#         class_cond,
#         semantic_segmentation_map_cond,
#         learn_sigma,
#         num_channels,
#         num_res_blocks,
#         channel_mult,
#         num_heads,
#         num_head_channels,
#         num_heads_upsample,
#         attention_resolutions,
#         dropout,
#         diffusion_steps,
#         noise_schedule,
#         timestep_respacing,
#         use_kl,
#         predict_xstart,
#         rescale_timesteps,
#         rescale_learned_sigmas,
#         use_checkpoint,
#         use_scale_shift_norm,
#         resblock_updown,
#         use_fp16,
#         use_new_attention_order,
#         use_gan_loss,
#         num_classes,
#         num_classes_for_semantic_segmentation_map,
#         classifier_free,
#         in_channels,
#         layout_cond,
#         num_classes_for_layout_object,
#         mask_size_for_layout_object,
#         layout_length,
#         xf_width,
#         xf_layers,
#         xf_heads,
#         xf_final_ln,
#         reconstruct_object_image,
#         reconstruct_size,
#         attention_resolutions_to_reconstruct_object_image,
#         reconstruct_object_image_loss_weight
# ):
#     model = create_model(
#         image_size,
#         num_channels,
#         num_res_blocks,
#         channel_mult=channel_mult,
#         learn_sigma=learn_sigma,
#         class_cond=class_cond,
#         semantic_segmentation_map_cond=semantic_segmentation_map_cond,
#         layout_cond=layout_cond,
#         use_checkpoint=use_checkpoint,
#         attention_resolutions=attention_resolutions,
#         num_heads=num_heads,
#         num_head_channels=num_head_channels,
#         num_heads_upsample=num_heads_upsample,
#         use_scale_shift_norm=use_scale_shift_norm,
#         dropout=dropout,
#         resblock_updown=resblock_updown,
#         use_fp16=use_fp16,
#         use_new_attention_order=use_new_attention_order,
#         num_classes=num_classes,
#         num_classes_for_semantic_segmentation_map=num_classes_for_semantic_segmentation_map,
#         classifier_free=classifier_free,
#         in_channels=in_channels,
#         num_classes_for_layout_object=num_classes_for_layout_object,
#         mask_size_for_layout_object=mask_size_for_layout_object,
#         layout_length=layout_length,
#         xf_width=xf_width,
#         xf_layers=xf_layers,
#         xf_heads=xf_heads,
#         xf_final_ln=xf_final_ln,
#         reconstruct_object_image=reconstruct_object_image,
#         reconstruct_size=reconstruct_size,
#         attention_resolutions_to_reconstruct_object_image=attention_resolutions_to_reconstruct_object_image
#     )
#     diffusion = create_gaussian_diffusion(
#         steps=diffusion_steps,
#         learn_sigma=learn_sigma,
#         noise_schedule=noise_schedule,
#         use_kl=use_kl,
#         predict_xstart=predict_xstart,
#         rescale_timesteps=rescale_timesteps,
#         rescale_learned_sigmas=rescale_learned_sigmas,
#         timestep_respacing=timestep_respacing,
#         reconstruct_object_image_loss_weight=reconstruct_object_image_loss_weight
#     )
#     return model, diffusion
#
#
# def create_model(
#         image_size,
#         num_channels,
#         num_res_blocks,
#         channel_mult="",
#         learn_sigma=False,
#         class_cond=False,
#         semantic_segmentation_map_cond=False,
#         use_checkpoint=False,
#         attention_resolutions="16",
#         num_heads=1,
#         num_head_channels=-1,
#         num_heads_upsample=-1,
#         use_scale_shift_norm=False,
#         dropout=0,
#         resblock_updown=False,
#         use_fp16=False,
#         use_new_attention_order=False,
#         num_classes=1000,
#         num_classes_for_semantic_segmentation_map=182,
#         classifier_free=False,
#         in_channels=3,
#         layout_cond=False,
#         num_classes_for_layout_object=185,
#         mask_size_for_layout_object=256,
#         layout_length=10,
#         xf_width=512,
#         xf_layers=16,
#         xf_heads=8,
#         xf_final_ln=True,
#         reconstruct_object_image=False,
#         reconstruct_size=32,
#         attention_resolutions_to_reconstruct_object_image='32'
# ):
#     if channel_mult == "":
#         if image_size == 512:
#             channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
#         elif image_size == 256:
#             channel_mult = (1, 1, 2, 2, 4, 4)
#         elif image_size == 128:
#             channel_mult = (1, 1, 2, 3, 4)
#         elif image_size == 64:
#             channel_mult = (1, 2, 3, 4)
#         elif image_size == 32:
#             channel_mult = (1, 2, 2, 2)
#         else:
#             raise ValueError(f"unsupported image size: {image_size}")
#     else:
#         channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
#
#     attention_ds = []
#     for res in attention_resolutions.split(","):
#         attention_ds.append(image_size // int(res))
#
#     if layout_cond:
#         return Layout2ImUNet(
#             image_size=image_size,
#             in_channels=in_channels,
#             model_channels=num_channels,
#             out_channels=(3 if not learn_sigma else 6),
#             num_res_blocks=num_res_blocks,
#             attention_resolutions=tuple(attention_ds),
#             dropout=dropout,
#             channel_mult=channel_mult,
#             num_classes=(num_classes if class_cond else None),
#             num_classes_for_semantic_segmentation_map=(num_classes_for_semantic_segmentation_map if semantic_segmentation_map_cond else None),
#             use_checkpoint=use_checkpoint,
#             use_fp16=use_fp16,
#             num_heads=num_heads,
#             num_head_channels=num_head_channels,
#             num_heads_upsample=num_heads_upsample,
#             use_scale_shift_norm=use_scale_shift_norm,
#             resblock_updown=resblock_updown,
#             use_new_attention_order=use_new_attention_order,
#             classifier_free=classifier_free,
#             num_classes_for_layout_object=num_classes_for_layout_object,
#             mask_size_for_layout_object=mask_size_for_layout_object,
#             layout_length=layout_length,
#             xf_width=xf_width,
#             xf_layers=xf_layers,
#             xf_heads=xf_heads,
#             xf_final_ln=xf_final_ln,
#             reconstruct_object_image=reconstruct_object_image,
#             reconstruct_size=reconstruct_size,
#             attention_resolutions_to_reconstruct_object_image=tuple(int(resolution) for resolution in attention_resolutions_to_reconstruct_object_image.split(','))
#         )
#     else:
#         return UNetModel(
#             image_size=image_size,
#             in_channels=in_channels,
#             model_channels=num_channels,
#             out_channels=(3 if not learn_sigma else 6),
#             num_res_blocks=num_res_blocks,
#             attention_resolutions=tuple(attention_ds),
#             dropout=dropout,
#             channel_mult=channel_mult,
#             num_classes=(num_classes if class_cond else None),
#             num_classes_for_semantic_segmentation_map=(num_classes_for_semantic_segmentation_map if semantic_segmentation_map_cond else None),
#             use_checkpoint=use_checkpoint,
#             use_fp16=use_fp16,
#             num_heads=num_heads,
#             num_head_channels=num_head_channels,
#             num_heads_upsample=num_heads_upsample,
#             use_scale_shift_norm=use_scale_shift_norm,
#             resblock_updown=resblock_updown,
#             use_new_attention_order=use_new_attention_order,
#             classifier_free=classifier_free,
#         )
#
#
# def create_classifier_and_diffusion(
#         image_size,
#         classifier_use_fp16,
#         classifier_width,
#         classifier_depth,
#         classifier_attention_resolutions,
#         classifier_use_scale_shift_norm,
#         classifier_resblock_updown,
#         classifier_pool,
#         learn_sigma,
#         diffusion_steps,
#         noise_schedule,
#         timestep_respacing,
#         use_kl,
#         predict_xstart,
#         rescale_timesteps,
#         rescale_learned_sigmas,
#         classifier_out_channel,
#         use_gan_loss,
#         num_classes,
#         dropout,
#         classifier_num_head_channels,
#         reconstruct_object_image_loss_weight
# ):
#     classifier = create_classifier(
#         image_size,
#         classifier_use_fp16,
#         classifier_width,
#         classifier_depth,
#         classifier_attention_resolutions,
#         classifier_use_scale_shift_norm,
#         classifier_resblock_updown,
#         classifier_pool,
#         use_MCD,
#         use_hierarchical_unet_classifiers,
#         max_hierarchical_level,
#         use_checknet,
#         classifier_out_channel,
#         num_classes,
#         dropout,
#         classifier_num_head_channels
#     )
#     diffusion = create_gaussian_diffusion(
#         steps=diffusion_steps,
#         learn_sigma=learn_sigma,
#         noise_schedule=noise_schedule,
#         use_kl=use_kl,
#         predict_xstart=predict_xstart,
#         rescale_timesteps=rescale_timesteps,
#         rescale_learned_sigmas=rescale_learned_sigmas,
#         timestep_respacing=timestep_respacing,
#         use_entropy_scale=use_entropy_scale,
#         scale_object=scale_object,
#         use_gan_loss=use_gan_loss,
#         reconstruct_object_image_loss_weight=reconstruct_object_image_loss_weight
#     )
#     return classifier, diffusion
#
#
# def create_classifier(
#         image_size,
#         classifier_use_fp16,
#         classifier_width,
#         classifier_depth,
#         classifier_attention_resolutions,
#         classifier_use_scale_shift_norm,
#         classifier_resblock_updown,
#         classifier_pool,
#         use_MCD,
#         use_hierarchical_unet_classifiers,
#         max_hierarchical_level,
#         use_checknet,
#         classifier_out_channel,
#         num_classes,
#         dropout,
#         classifier_num_head_channels
# ):
#     if image_size == 512:
#         channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
#     elif image_size == 256:
#         channel_mult = (1, 1, 2, 2, 4, 4)
#     elif image_size == 128:
#         channel_mult = (1, 1, 2, 3, 4)
#     elif image_size == 64:
#         channel_mult = (1, 2, 3, 4)
#     elif image_size == 32:
#         channel_mult = (1, 2, 2, 2)
#     else:
#         raise ValueError(f"unsupported image size: {image_size}")
#
#     attention_ds = []
#     for res in classifier_attention_resolutions.split(","):
#         attention_ds.append(image_size // int(res))
#
#     if use_hierarchical_unet_classifiers:
#         return HierarchicalUNetClassifier(
#             image_size=image_size,
#             in_channels=3,
#             model_channels=classifier_width,
#             out_channels=1000,
#             num_res_blocks=classifier_depth,
#             attention_resolutions=tuple(attention_ds),
#             channel_mult=channel_mult,
#             use_fp16=classifier_use_fp16,
#             num_head_channels=classifier_num_head_channels,
#             use_scale_shift_norm=classifier_use_scale_shift_norm,
#             resblock_updown=classifier_resblock_updown,
#             max_hierarchical_level=max_hierarchical_level,
#             dropout=dropout
#         )
#
#     if use_checknet:
#         return CheckNet(
#             image_size=image_size,
#             in_channels=3,
#             model_channels=classifier_width,
#             out_channels=1,
#             num_res_blocks=classifier_depth,
#             attention_resolutions=tuple(attention_ds),
#             channel_mult=channel_mult,
#             use_fp16=classifier_use_fp16,
#             num_head_channels=classifier_num_head_channels,
#             use_scale_shift_norm=classifier_use_scale_shift_norm,
#             resblock_updown=classifier_resblock_updown,
#             num_classes=1000,
#             dropout=dropout
#         )
#
#     return EncoderUNetModel(
#         image_size=image_size,
#         in_channels=3,
#         model_channels=classifier_width,
#         out_channels=classifier_out_channel,
#         num_res_blocks=classifier_depth,
#         attention_resolutions=tuple(attention_ds),
#         channel_mult=channel_mult,
#         use_fp16=classifier_use_fp16,
#         num_head_channels=classifier_num_head_channels,
#         use_scale_shift_norm=classifier_use_scale_shift_norm,
#         resblock_updown=classifier_resblock_updown,
#         pool=classifier_pool,
#         use_MCD=use_MCD,
#         dropout=dropout
#     )
#
#
#
#
#
#
#
# def create_gaussian_diffusion(
#         *,
#         steps=1000,
#         learn_sigma=False,
#         sigma_small=False,
#         noise_schedule="linear",
#         use_kl=False,
#         predict_xstart=False,
#         rescale_timesteps=False,
#         rescale_learned_sigmas=False,
#         timestep_respacing="",
#         use_entropy_scale=False,
#         use_gan_loss=False,
#         reconstruct_object_image_loss_weight=0.0
# ):
#     betas = gd.get_named_beta_schedule(noise_schedule, steps)
#     if use_kl:
#         loss_type = 'RESCALED_KL'
#     elif rescale_learned_sigmas:
#         loss_type = 'RESCALED_MSE'
#         if use_gan_loss:
#             loss_type = 'RESCALED_MSE_and_GAN'
#         if reconstruct_object_image_loss_weight:
#             loss_type = 'RESCALED_MSE_RECONSTRUCT_OBJECT_IMAGE'
#     else:
#         loss_type = 'MSE'
#
#     if not timestep_respacing:
#         timestep_respacing = [steps]
#     return SpacedDiffusion(
#         use_timesteps=space_timesteps(steps, timestep_respacing),
#         betas=betas,
#         model_mean_type=(
#             'EPSILON' if not predict_xstart else 'START_X'
#         ),
#         model_var_type=(
#             (
#                 'FIXED_LARGE'
#                 if not sigma_small
#                 else 'FIXED_SMALL'
#             )
#             if not learn_sigma
#             else 'LEARNED_RANGE'
#         ),
#         loss_type=loss_type,
#         rescale_timesteps=rescale_timesteps,
#         use_entropy_scale=use_entropy_scale,
#         reconstruct_object_image_loss_weight=reconstruct_object_image_loss_weight
#     )
#
#
# def add_dict_to_argparser(parser, default_dict):
#     for k, v in default_dict.items():
#         v_type = type(v)
#         if v is None:
#             v_type = str
#         elif isinstance(v, bool):
#             v_type = str2bool
#         parser.add_argument(f"--{k}", default=v, type=v_type)
#
#
# def args_to_dict(args, keys):
#     return {k: getattr(args, k) for k in keys}
#
#
# def str2bool(v):
#     """
#     https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
#     """
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ("yes", "true", "t", "y", "1"):
#         return True
#     elif v.lower() in ("no", "false", "f", "n", "0"):
#         return False
#     else:
#         raise argparse.ArgumentTypeError("boolean value expected")
