
cd /workspace/mnt/storage/guangcongzheng/zju_zgc/LayoutDiffusion
conda activate

pip install omegaconf

rm -rf /opt/conda/lib/python3.8/site-packages/layout_diffusion

rm -rf ./layout_diffusion.egg-info

python setup.py build develop

WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=$((RANDOM + 10000))


# pip install  tensorflow-gpu==2.7.0 blobfile  mpi4py tqdm requests pandas
#     conda install -y openmpi
#
#WORKSPACE = "/workspace/guided-diffusion/pretrain_model"
#
#cd $WORKSPACE
#wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt
#wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt
#wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
#
