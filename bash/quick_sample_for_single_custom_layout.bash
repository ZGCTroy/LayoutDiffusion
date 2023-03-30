cd /workspace/mnt/storage/guangcongzheng/zju_zgc/LayoutDiffusion
conda activate

pip install omegaconf
pip install -e ../dpm_solver
rm -rf /opt/conda/lib/python3.8/site-packages/layout_diffusion

rm -rf ./layout_diffusion.egg-info

python setup.py build develop

cp /workspace/mnt/storage/guangcongzheng/zju_zgc/TTUR/classify_image_graph_def.pb /tmp/classify_image_graph_def.pb

WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=$((RANDOM + 10000))

VERSION='LayoutDiffusion-v7_large'
SAMPLE_METHOD='dpm_solver' # ['dpm_solver', 'ddpm', 'ddim']
CLASSIFIER_SCALE=1.0
DATASET='COCO-stuff' # ['COCO-stuff', 'VG']
IMAGE_SIZE=256
CHECKPOINT='1000000'
STEPS=25
SAMPLE_TIMES=1
SAMPLE_ROOT='/workspace/mnt/storage/3150104097/zju_tiny_backup/samples'
#SAMPLE_ROOT='/workspace/mnt/storage/guangcongzheng/zju_zgc_backup/samples'

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch  \
      --nnodes $WORLD_SIZE \
      --node_rank $RANK \
      --nproc_per_node 1 \
      --master_addr $MASTER_ADDR \
      --master_port $MASTER_PORT \
      scripts/classifier-free_sample_for_single_custom_layout.py  \
      --config_file ./configs/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/${VERSION}.yaml \
       sample.pretrained_model_path=./log/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/${VERSION}/ema_0.9999_${CHECKPOINT}.pt \
       sample.log_root=${SAMPLE_ROOT}/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/${VERSION} \
       sample.timestep_respacing=[${STEPS}] \
       sample.sample_suffix=model${CHECKPOINT}_scale${CLASSIFIER_SCALE}_${SAMPLE_METHOD}_single_custom_layout  \
       sample.classifier_free_scale=${CLASSIFIER_SCALE} \
       sample.sample_method=${SAMPLE_METHOD} \
       sample.sample_times=${SAMPLE_TIMES} \
       sample.save_images_with_bboxs=True \
       sample.save_cropped_images=True


