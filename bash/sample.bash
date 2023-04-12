WORKSPACE='/workspace/mnt/storage/guangcongzheng/zju_zgc/LayoutDiffusion'
cd ${WORKSPACE}

SAMPLE_ROOT='/workspace/mnt/storage/guangcongzheng/zju_zgc_backup/samples'

VERSION='LayoutDiffusion_large'
MODEL_ITERATION='1150000'

DATASET='COCO-stuff' # ['COCO-stuff', 'VG']
IMAGE_SIZE=256

SAMPLE_METHOD='dpm_solver' # ['dpm_solver', 'ddpm', 'ddim']
STEPS=25
SAMPLE_TIMES=5
NUM_IMG=2048 # 2048, 3097, 5096


CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch  \
      --nproc_per_node 2 \
      scripts/classifier-free_sample_for_layout.py  \
      --config_file ./configs/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/${VERSION}.yaml \
       sample.pretrained_model_path=./pretrained_models/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}_${VERSION}_ema_${MODEL_ITERATION} \
      sample.log_root=${SAMPLE_ROOT}/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/${VERSION} \
      sample.sample_method=${SAMPLE_METHOD} sample.timestep_respacing=[${STEPS}] sample.classifier_free_scale=${CLASSIFIER_SCALE} sample.sample_times=${SAMPLE_TIMES} \
      sample.sample_suffix=model${MODEL_ITERATION}_scale${CLASSIFIER_SCALE}_${SAMPLE_METHOD} \
      sample.save_cropped_images=True sample.fix_seed=False \
      data.parameters.test.deprecated_stuff_ids_txt='./layout_diffusion/dataset/image_ids_layout2i_2048.txt'  # same as Frido/frido/data/image_ids_layout2i_2048.txt, only used for COCO-stuff 2048x1
#      data.parameters.test.max_num_samples=${NUM_IMG} \    # used for 3097 x 5
#      data.parameters.filter_mode=SG2Im \


