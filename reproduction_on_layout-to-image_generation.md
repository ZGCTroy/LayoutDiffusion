https://paperswithcode.com/task/layout-to-image-generation

https://lists.papersapp.com/EfLwCOleijHg

## 要点

- Scene-to-Image Generation, 鼻祖是SG2Im, 输入scene graph预测得到fake的[obj_class, obj_box, obj_class]。 但是可以在sample时直接输入gt的[obj_class, obj_box, obj_class]转换成
layout-to-image generation，但是要注意输入中是有obj_mask的，不太公平。

- 我们做的是Layout-to-Image Generation，鼻祖是Layout2Im，sample时输入是obj序列: [obj_class, obj_box]，没有obj_mask。 要注意某些方法可能在训练时使用了obj_mask，但是
sample时输入不一定输入了mask，要留意一下。

- 数据集做[COCO-stuff](https://github.com/nightrome/cocostuff) 和 [Visual Genome](https://visualgenome.org/)
  - coco-stuff分为已经被deprecate的segmentation coco-stuff 和 全量的 full coco-stuff， 后续两种coco都会做， config里用cfg.data.parameters.use_deprecated_stuff2017: True 代表使用小coco
  - vg的SG2Im原论文说62565 train, 5506 val, 5088 test， 而我使用他公布的vg_splits.json， download_vg.sh 和 preprocess_vg.py 得到
  的是62565 train, 5062 val, 5096 test, 只有train是符合的。比较新的AttrLostGAN跟我得到了同样的结果，小概率是版本问题。所以生成vg数据集时，记得严格遵守SG2Im的环境，再分割vg数据集（我已经试过了，没用，你可以在2080上试试）.

- samples. 按顺序复现。torch 版本关注一下公开的requirements.txt 和 issues里作者的回答，一般都是torch1.0, 或<= torch1.5。
  - 关注torch 版本，requirements.txt不一定准，要看issues里
  - 先复现SG2Im, conda create -n SG2Im, torch 可能是1.0.0。 
  - 复现Layout2Im, conda create -n Layout2Im
  - 复现Grid2Im, conda create -n Grid2Im
  - 复现LostGAN, conda create -n LostGAN
  - 生成png或者jpg会有影响，复现时要注意，可以看[AttrLostGAN](https://www.readcube.com/library/e42cbdf5-6f60-4c70-bd9f-911cf106a59c:f616b2cc-536e-428c-b6ba-4159f12e00b5) 的论文的补充材料,第11页有详细解释。
  - cfg.sample.sample_times=5, 表示generated imgs生成5份，保存成filename_{idx}.png, idx为第几份。而real imgs只有1份。
  - coco-stuff下，cfg.data.parameters.filter_mode='SG2Im' 则是test=2048， filter_mode='LostGAN'则是全部的test=3097
  - vg sample只用test集，目前是5096,先复现看看跟论文的5088差别大不大，一般是每个生成5份，某些人只生成了1份
  
- 测试上可以参考的有[LostGAN-v2的FID, IS, LPIPS, CAS](https://github.com/WillSuen/LostGANs/issues/3) 和 [Grid2Im的IS, FID, LPIPS, CAS](https://github.com/ashual/scene_generation) 和 
  [AttrLostGAN的IS, SceneIS, FID, SceneFID, LPIPS, CAS](https://github.com/stanifrolov/AttrLostGAN)
  - [TTUR](https://github.com/bioinf-jku/TTUR) 测FID和SceneFid，conda create -n TTUR, 先尝试tensorflow 1.15，不行再< 1.15，及时汇报情况。 服务器上的TTUR_v2是可以run起来的，用的是tensorflow 2.7。
  SceneFid要把生成的图片读取然后crop出每个object保存成图片，可以参考[LAMA的utils/extract_cropped_objects.py](https://github.com/ZejianLi/LAMA) 。
  也可以在classifier_sample时crop，不仅保存real imgs，fake imgs还保存 real crop imgs 和 fake crop imgs
  - [improved-gan](https://github.com/openai/improved-gan) 测IS, conda create -n improved-gan, 使用tensorflow来测
  - [LPIPS](https://github.com/richzhang/PerceptualSimilarity) 测 LPIPS作为Diversity Score， conda create -n LPIPS, pytorch环境
  - 测CAS， classification
  

## 1. Stanford, Google AI, LiFeiFei

### 1.1. SG2Im, 2018CVPR

* [Image Generation from Scene Graphs](https://www.readcube.com/library/e42cbdf5-6f60-4c70-bd9f-911cf106a59c:47691753-2071-42cc-99ed-f68b642dfd0d)
* [github](https://github.com/google/sg2im)
* Segmentation COCO: the 2017 COCO Thing and Stuff Segmentation Challenge* subset (50k images). It features bounding boxes for 80 object and 91 stuff classes (excluding “other”). We filter for images
  with 3 to 8 objects, each larger than 2 % of the image’s area. Obtain 24972 training, 1024 validation and 2048 test images. When evaluating, generate 5 images for each test image, 2048x5=10240.

  
## 2. University of British Columbia Vector Institute, Leonid Sigal组
仅复现期刊版本的Layout2Im

### 2.1. Layout2Im, 2019CVPR

* [Image Generation from Layout](https://www.readcube.com/library/e42cbdf5-6f60-4c70-bd9f-911cf106a59c:eb822f13-f18d-490f-829c-73ac04c7c0f7)
* 使用了[TTUR](https://github.com/bioinf-jku/TTUR)计算FID. IS使用[code](https://github.com/zhaobozb/layout2im/files/3572029/compute.inception.score.zip)

### 2.2. Layout2image, 2020IJCV

* [Layout2image: Image Generation from Layout](https://www.readcube.com/library/e42cbdf5-6f60-4c70-bd9f-911cf106a59c:208366d8-ac1d-48b2-8c92-42ee319ff6ba)
* [github](https://github.com/google/sg2im)
* 使用了[TTUR](https://github.com/bioinf-jku/TTUR)计算FID. IS使用[code](https://github.com/zhaobozb/layout2im/files/3572029/compute.inception.score.zip)

## 3. Tel Aviv University and Facebook AI Research

### 3.1. Grid2Im, 2019ICCV

* [Specifying Object Attributes and Relations in Interactive Scene Generation](https://www.readcube.com/library/e42cbdf5-6f60-4c70-bd9f-911cf106a59c:a82cae62-856a-4450-9bb7-7683172c8ab1)
* [github](https://github.com/ashual/scene_generation)

## 4. North Carolina State University, Tianfu Wu组

### 4.1. LostGANs-v1, 2019ICCV

* [Image Synthesis From Reconfigurable Layout and Style](https://www.readcube.com/library/e42cbdf5-6f60-4c70-bd9f-911cf106a59c:2ac7921f-5601-4f3b-b1cc-f0e5579c5e58)

### 4.2. LostGANs-v2, 2021 TPAMI

* [Learning Layout and Style Reconfigurable GANs for Controllable Image Synthesis](https://www.readcube.com/library/e42cbdf5-6f60-4c70-bd9f-911cf106a59c:12b7abc3-3ccc-452e-b9c2-7985c4718c74)

* [github](https://github.com/WillSuen/LostGANs)

* [FID, IS, DS, CAS](https://github.com/WillSuen/LostGANs/issues/3)

## 5. Mila, Montr ́eal, Canada

### 5.1. OC-GAN, 2021AAAI

* [Object-Centric Image Generation from Layouts](https://www.readcube.com/library/e42cbdf5-6f60-4c70-bd9f-911cf106a59c:668d3a3c-ebb6-4b51-97c1-404763b6ae92)
* 没有开源代码，不复现，最多引用论文内容

## 6. Huawei Technologies, Peng Du组

### 6.1. PLG, 2022CVPR

* [Interactive Image Synthesis with Panoptic Layout Generation](https://www.readcube.com/library/e42cbdf5-6f60-4c70-bd9f-911cf106a59c:e22a3cca-d054-400e-82e1-1c92ce78b606)

* [github](https://github.com/wb-finalking/PLGAN)

## 7. Machine Vision & Learning Group at the Ludwig Maximilian University of Munich (formerly the Computer Vision Group, Heidelberg University), Ommer-lab.

### 7.1. Taming-transformers

* [High-Resolution Complex Scene Synthesis with Transformers](https://www.readcube.com/library/e42cbdf5-6f60-4c70-bd9f-911cf106a59c:f6926abf-780c-4336-a684-28be81e5c024)

* 首次使用Full COCO: the full COCO Thing and Stuff dataset (164K images). When filtering for 3 to 8 objects, one gets 74121 training images. For comparability, validation is however done on the same 2048
  test images as above. Broadening the training exercise and filtering for 2 to 30 objects of any size, one obtains 112k training images.

* FID 使用的[toshas/torch-fidelit](https://github.com/toshas/torch-fidelity), 没有汇报IS

* 提出训练时数据增强可以增加性能，首次使用了全量的Full COCO，并首次在flip的基础上再加了random crop。性能提升均来源于这两个额外的数据增强。

* ![img.png](figures/taming-transformers.png)

### 7.2. LDM

* [2022 CVPR, High-Resolution Image Synthesis with Latent Diffusion Models](https://www.readcube.com/library/e42cbdf5-6f60-4c70-bd9f-911cf106a59c:441e05d3-76ce-46bf-983b-c69f90ff2e54)

* FID 使用的[toshas/torch-fidelit](https://github.com/toshas/torch-fidelity), 没有汇报IS

## 8. [2021 CVPR, Context-Aware Layout to Image Generation with Enhanced Object Appearance](https://www.readcube.com/library/e42cbdf5-6f60-4c70-bd9f-911cf106a59c:76b6b16a-4bf4-4967-80bf-10873fc5a618)

https://www.researchgate.net/publication/355733291_Image_Synthesis_from_Layout_with_Locality-Aware_Mask_Adaption

## 2. Experiments

* COCO-stuff 256x256

|                       | FID     | IS            |
|-----------------------|---------|---------------|
| Grid2Im (2019ICCV)    | 65.95   | 15.2 +- 0.1   |
| LostGAN-V2 (2021TPAMI)| 42.65   | 18.2 +- 0.2   |
| LDM (2022CVPR)        | 40.91   |    /          |
| CAL2I+PLG (2022CVPR)  | 29.1    | 18.9 +- 0.3   |
| Ours (LayoutDiffusion)| 18.33   | 24.09 +- 0.83 |

  
