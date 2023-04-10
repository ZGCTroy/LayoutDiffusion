WORKSPACE=/workspace/mnt/storage/guangcongzheng/zju_zgc/LayoutDiffusion
cd ${WORKSPACE}

conda create -n LayoutDiffusion python=3.8
conda activate LayoutDiffusion
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install imageio==2.9.0
pip install omegaconf opencv-python h5py==3.2.1 gradio==3.24.1
pip install -e ./repositories/dpm_solver


python setup.py build develop