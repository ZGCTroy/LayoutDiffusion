
WORKSPACE=/workspace/mnt/storage/guangcongzheng/zju_zgc/LayoutDiffusion
cd ${WORKSPACE}

conda activate LayoutDiffusion

python -m torch.distributed.launch \
       --nproc_per_node 2 \
       scripts/image_train_for_layout.py \
       --config_file ./configs/COCO-stuff_256x256/latent_LayoutDiffusion_large.yaml