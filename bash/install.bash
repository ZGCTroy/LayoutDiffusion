WORKSPACE=/workspace/mnt/storage/guangcongzheng/zju_zgc/test_for_github/LayoutDiffusion
cd ${WORKSPACE}

conda create -y -n LayoutDiffusion python=3.8

conda activate LayoutDiffusion

conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

conda install -y imageio==2.9.0

pip install omegaconf opencv-python h5py==3.2.1 gradio==3.38.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install -e ./repositories/dpm_solver

pip install --upgrade diffusers[torch]

python setup.py build develop


python scripts/launch_gradio_app.py  \
  --config_file configs/COCO-stuff_256x256/LayoutDiffusion_large.yaml \
  sample.pretrained_model_path=/workspace/mnt/storage/guangcongzheng/zju_zgc_backup/LayoutDiffusion/log/COCO-stuff_256x256/LayoutDiffusion-v7_large/ema_0.9999_1150000.pt --share