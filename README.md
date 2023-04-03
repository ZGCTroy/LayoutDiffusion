# News and ToDo List

- [ ] Improve README and code usage instructions
- [ ] Clean up code 
- [ ] Release pre-trained model
- [ ] Release tools for evaluation
- [x] 2023-03-30: Publish complete code 
- [x] 2023-02-27: Accepted by CVPR2023 
- [x] 2022-11-11: Submitted to CVPR2023 
- [x] 2022-07-08: Publish initial code


# Introduction
This repository is the official implementation of CVPR2023: [LayoutDiffusion: Controllable Diffusion Model for Layout-to-image Generation](https://arxiv.org/abs/2303.17189)

The code is heavily based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), 
with the following modifications:
1. Added support for Distributed Training of PyTorch.
2. Added support for OmegaConfig in ./configs for easy control
3. Added support for layout-to-image generation by introducing a layout encoder (layout fusion module or LFM) and object-aware cross-attention (OaCA).


* pipeline
![pipeline](./figures/pipeline.png)

* comparision with other methods on COCO
![compare_with_other_methods_on_COCO](./figures/comapre_with_other_methods_on_COCO.png)


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

  

# Cite
```
@misc{zheng2023layoutdiffusion,
    title={LayoutDiffusion: Controllable Diffusion Model for Layout-to-image Generation}, 
    author={Guangcong Zheng and Xianpan Zhou and Xuewei Li and Zhongang Qi and Ying Shan and Xi Li},
    year={2023},
    eprint={2303.17189},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
