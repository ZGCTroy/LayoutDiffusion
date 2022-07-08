
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

nohup python -m torch.distributed.launch \
       --nnodes $WORLD_SIZE \
       --node_rank $RANK \
       --nproc_per_node 8 \
       --master_addr $MASTER_ADDR \
       --master_port $MASTER_PORT \
       scripts/image_train_for_layout.py \
       --config_file ./configs/COCO-stuff_256x256/LayoutDiffusion-v1.yaml \
       > log/train_COCO-stuff_256x256_LayoutDiffusion-v1.txt
