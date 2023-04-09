# News and ToDo List

- [ ] Improve README and code usage instructions
- [ ] Clean up code 
- [ ] Release tools for evaluation
- [x] 2023-04-09: Release [pre-trained model](https://drive.google.com/drive/folders/1sJxbhi_pioFaHKgAAAuo8wZLIBuLbyxz?usp=sharing) 
- [x] 2023-04-09: Release instructions for setuping up environment and training 
- [x] 2023-04-09: Release Gradio Webui Demo
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

## Gradio Webui Demo
![pipeline](./figures/gradio_demo.png)

## Pipeline
![pipeline](./figures/pipeline.png)

## Visualizations on COCO-stuff
![compare_with_other_methods_on_COCO](./figures/comapre_with_other_methods_on_COCO.png)


# Setup Environment
```bash
conda create -n LayoutDiffusion python=3.8
conda activate LayoutDiffusion

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install omegaconf opencv-python h5py==3.2.1 gradio==3.24.1
pip install -e ./repositories/dpm_solver

python setup.py build develop
```

# Gradio Webui Demo
```bash
  python scripts/launch_gradio_app.py  \
  --config_file configs/COCO-stuff_256x256/LayoutDiffusion-v7_large.yaml \
  sample.pretrained_model_path=./log/COCO-stuff_256x256/LayoutDiffusion-v7_large/ema_0.9999_1150000.pt
```

# Training
* bash/train.bash
```bash
python -m torch.distributed.launch \
       --nproc_per_node 8 \
       scripts/image_train_for_layout.py \
       --config_file ./configs/COCO-stuff_256x256/LayoutDiffusion_large.yaml
```

## Evaluation (To be continued)

- Sampling for the entire validation dataset: 
  - Refer to 'bash/sample.bash'. 
  - Currently, the recommended settings are steps=25 and sample_method='dpm_solver'. 
  - When enabling full sampling, set 'cfg.sample.save_cropped_imgs' to True and 'cfg.sample.fix_seed' to False. 

- quick sample
  - Three sampling methods are available: ddpm, ddim, and dpm_solver. 
  - For selecting images of medium quality, it is recommended to use steps=25 and sample_method='dpm_solver'. 
  - For selecting images of high quality, it is recommended to use steps=200 and sample_method='ddim'. For high-quality images with steps=200 or 1000, it is recommended to use sample_method='ddpm'. 
  - The classifier scale should be set to 0.7, with an adjustable range of +- 0.5. 
  - Set cfg.sample.fix_seed=True, cfg.sample.save_cropped_images=False, cfg.sample.save_images_with_bboxs=True, and cfg.sample.save_sequence_of_obj_imgs=True. 
  - For data sampling, set data.parameters.test.max_num_samples=64 to select the first 64 images in the data set, and data.parameters.test.batch_size=8. 
  - To select specific image IDs for COCO-stuff, set data.parameters.test.specific_image_ids='['VG_100K_2/103.jpg', 'VG_100K_2/113.jpg']', with specific_image_ids having a higher priority than max_num_samples. To select image IDs for VG, set data.parameters.test.specific_image_ids='[87038, 174482]'.


  

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
