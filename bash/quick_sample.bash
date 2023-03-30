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
CLASSIFIER_SCALE=0.6
DATASET='COCO-stuff' # ['COCO-stuff', 'VG']
IMAGE_SIZE=128
CHECKPOINT='0300000'
STEPS=25
SAMPLE_TIMES=1
SAMPLE_ROOT='/workspace/mnt/storage/3150104097/zju_tiny_backup/samples'
#SAMPLE_ROOT='/workspace/mnt/storage/guangcongzheng/zju_zgc_backup/samples'

python
import torch
x = torch.Tensor([1.0])
x.to('cuda:0')
x.to('cuda:1')
exit()

CUDA_VISIBLE_DEVICES=1 \
python -m torch.distributed.launch  \
      --nnodes $WORLD_SIZE \
      --node_rank $RANK \
      --nproc_per_node 2 \
      --master_addr $MASTER_ADDR \
      --master_port $MASTER_PORT \
      scripts/classifier-free_sample_for_layout.py  \
      --config_file ./configs/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/${VERSION}.yaml \
       sample.pretrained_model_path=./log/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/${VERSION}/ema_0.9999_${CHECKPOINT}.pt \
       sample.log_root=${SAMPLE_ROOT}/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/${VERSION} \
       sample.timestep_respacing=[${STEPS}] \
       sample.sample_suffix=model${CHECKPOINT}_scale${CLASSIFIER_SCALE}_${SAMPLE_METHOD}  \
       sample.classifier_free_scale=${CLASSIFIER_SCALE} \
       sample.sample_method=${SAMPLE_METHOD} \
       sample.sample_times=${SAMPLE_TIMES} \
       data.parameters.test.max_num_samples=64 data.parameters.test.batch_size=8 \
       sample.save_images_with_bboxs=True \
       sample.save_sequence_of_obj_imgs=True \
#       data.parameters.test.specific_image_ids='['VG_100K_2/103.jpg', 'VG_100K_2/113.jpg']'
#        data.parameters.test.specific_image_ids='[87038, 174482]'
#       sample.save_cropped_images=True sample.fix_seed=False
#       model.parameters.ds_to_return_attention_embeddings=[4]

