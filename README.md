# Introduction
This repository is the official implementation of CVPR2023: LayoutDiffusion: Controllable Diffusion Model for Layout-to-image Generation

The code is heavily based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), 
with the following modifications: 
1. Added support for Distributed Training of PyTorch. 
2. Added support for layout-to-image generation by introducing a layout encoder (layout fusion module or LFM) and object-aware cross-attention (OaCA). 

# News and ToDo List 
- [ ] Improve README and code usage instructions
- [ ] Clean up code 
- [ ] Release pre-trained model
- [ ] Release tools for evaluation
- [x] 2023-03-10: Publish complete code 
- [x] 2023-02-27: Accepted by CVPR2023 
- [x] 2022-11-11: Submitted to CVPR2023 
- [x] 2022-07-08: Publish initial code

# Setup Environment
!!!  Coming soon  !!!

# Training
* bash/cmd1.bash

## Sampling
* bash/quick_sample.bash

- quick sample
  - Three sampling methods are available: ddpm, ddim, and dpm_solver. 
  - For selecting images of medium quality, it is recommended to use steps=25 and sample_method='dpm_solver'. 
  - For selecting images of high quality, it is recommended to use steps=200 and sample_method='ddim'. For high-quality images with steps=200 or 1000, it is recommended to use sample_method='ddpm'. 
  - The classifier scale should be set to 0.7, with an adjustable range of +- 0.5. 
  - Set cfg.sample.fix_seed=True, cfg.sample.save_cropped_images=False, cfg.sample.save_images_with_bboxs=True, and cfg.sample.save_sequence_of_obj_imgs=True. 
  - For data sampling, set data.parameters.test.max_num_samples=64 to select the first 64 images in the data set, and data.parameters.test.batch_size=8. 
  - To select specific image IDs for COCO-stuff, set data.parameters.test.specific_image_ids='['VG_100K_2/103.jpg', 'VG_100K_2/113.jpg']', with specific_image_ids having a higher priority than max_num_samples. To select image IDs for VG, set data.parameters.test.specific_image_ids='[87038, 174482]'.

- For custom layout sampling: 
  - Edit the 'custom_layout' dictionary at the beginning of the 'scripts/classifier-free_sample_for_single_custom_layout.py' file. 
  - Then use the command in 'quick_sample_for_single_custom_layout.bash' to generate the layout. 
 
- For full sampling: 
  - Refer to 'bash/sample.bash'. 
  - Currently, the recommended settings are steps=25 and sample_method='dpm_solver'. 
  - When enabling full sampling, set 'cfg.sample.save_cropped_imgs' to True and 'cfg.sample.fix_seed' to False. 


## 2. Experiments

* COCO-stuff 256x256

|                       | FID     | IS            |
|-----------------------|---------|---------------|
| Grid2Im (2019ICCV)    | 65.95   | 15.2 +- 0.1   |
| LostGAN-V2 (2021TPAMI)| 42.65   | 18.2 +- 0.2   |
| LDM (2022CVPR)        | 40.91   |    /          |
| CAL2I+PLG (2022CVPR)  | 29.1    | 18.9 +- 0.3   |
| Ours (LayoutDiffusion)| 18.33   | 24.09 +- 0.83 |

# Cite
```
@inproceedings{zheng2022entropy,
  title={Entropy-Driven Sampling and Training Scheme for Conditional Diffusion Generation},
  author={Zheng, Guangcong and Li, Shengming and Wang, Hui and Yao, Taiping and Chen, Yang and Ding, Shouhong and Li, Xi},
  booktitle={European Conference on Computer Vision},
  pages={754--769},
  year={2022},
  organization={Springer}
}
```
