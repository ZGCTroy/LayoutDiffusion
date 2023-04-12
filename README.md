## News and ToDo List

- [ ] Improve README and code usage instructions
- [ ] Clean up code 
- [ ] Release tools for evaluation
- [x] 2023-04-09: Release [pre-trained model](https://drive.google.com/drive/folders/1sJxbhi_pioFaHKgAAAuo8wZLIBuLbyxz?usp=sharing) 
- [x] 2023-04-09: Release instructions for environment and training 
- [x] 2023-04-09: Release Gradio Webui Demo
- [x] 2023-03-30: Publish complete code 
- [x] 2023-02-27: Accepted by CVPR2023 
- [x] 2022-11-11: Submitted to CVPR2023 
- [x] 2022-07-08: Publish initial code



## Introduction
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


## Setup Environment
```bash
conda create -n LayoutDiffusion python=3.8
conda activate LayoutDiffusion

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install omegaconf opencv-python h5py==3.2.1 gradio==3.24.1
pip install -e ./repositories/dpm_solver

python setup.py build develop
```

## Gradio Webui Demo (No need for setup of dataset)
```bash
  python scripts/launch_gradio_app.py  \
  --config_file configs/COCO-stuff_256x256/LayoutDiffusion_large.yaml \
  sample.pretrained_model_path=./pretrained_models/COCO-stuff_256x256_LayoutDiffusion_large_ema_1150000.pt
```

## Setup Dataset
See [here](./DATASET_SETUP.md)

## Pretrained Models
| Dataset                                                                                                                                   | Resolution |                   steps, FID (Sample imgs x times)                   | Link (TODO)                                                                                            |
|-------------------------------------------------------------------------------------------------------------------------------------------|:----------:|:--------------------------------------------------------------------:|--------------------------------------------------------------------------------------------------------|
| COCO-Stuff 2017 segmentation challenge<br/>([deprecated coco-stuff, not full coco-stuff](https://github.com/WillSuen/LostGANs/issues/19)) | 256 x 256  | steps=25 <br/> FID=15.61  ( 3097 x 5 ) <br/> FID=15.61  ( 2048 x 1 ) | [Google drive](https://drive.google.com/file/d/1aWIh-jPzNqXZibq8HlSeQfQzyXO8aMUK/view?usp=share_link)  | 
| COCO-Stuff 2017 segmentation challenge<br/>([deprecated coco-stuff, not full coco-stuff](https://github.com/WillSuen/LostGANs/issues/19)) | 128 x 128  |               steps=25 <br/>  FID=16.57  ( 3097 x 5 )                | [Google drive](https://drive.google.com/file/d/1LoNKfGabuXc53gh1FYGbVvwbJZjpjE3a/view?usp=share_link)  | 
| VG                                                                                                                                        | 256 x 256  |               steps=25 <br/>  FID=15.63  ( 5097 x 1 )                | [Google drive](https://drive.google.com/file/d/16CV4a-4e8gyzOemK8XP0j4KwNL8PGb1L/view?usp=share_link)  | 
| VG                                                                                                                                        | 128 x 128  |               steps=25 <br/>  FID=16.35  ( 5097 x 1 )                | [Google drive](https://drive.google.com/file/d/1NaC3oS9uG0DmgU8VgIDB-xESauczuAaV/view?usp=share_link)  | 

## Training
* bash/train.bash
```bash
python -m torch.distributed.launch \
       --nproc_per_node 8 \
       scripts/image_train_for_layout.py \
       --config_file ./configs/COCO-stuff_256x256/LayoutDiffusion_large.yaml
```

## Sampling
* bash/quick_sample.bash for quick sample
* bash/sample.bash for sample entire test dataset

## Evaluation (To be continued)


## For beginner
The field of layout-to-image generation is related to scenegraph-to-image generation and remained some confusing issues.
You could refer to issues like:
* [the deprecated coco-stuff 2017](https://github.com/WillSuen/LostGANs/issues/19)
* [FID, IS, LPIPS, CAS of LostGAN-v2](https://github.com/WillSuen/LostGANs/issues/3)
* [IS, FID, LPIPS, CAS of Grid2Im](https://github.com/ashual/scene_generation) 
* [IS, SceneIS, FID, SceneFID, LPIPS, CAS of AttrLostGAN](https://github.com/stanifrolov/AttrLostGAN)

However, it is recommended to ignore the confusing history and follow the latest [LDM](https://arxiv.org/pdf/2112.10752.pdf), [Frido](https://github.com/davidhalladay/Frido) to work on a relatively new benchmark.


## Cite
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
