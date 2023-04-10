WORKSPACE='/workspace/mnt/storage/guangcongzheng/zju_zgc/LayoutDiffusion'
cd ${WORKSPACE}

VERSION='LayoutDiffusion-v7_large'
SAMPLE_METHOD='dpm_solver' # ['dpm_solver', 'ddpm', 'ddim']
SAMPLE_TIMES=5
#STEPS=10
#SAMPLE_ROOT='/workspace/mnt/storage/3150104097/zju_tiny_backup/samples'
SAMPLE_ROOT='/workspace/mnt/storage/guangcongzheng/zju_zgc_backup/samples'

IMAGE_SIZE=128
DATASET='COCO-stuff' # ['COCO-stuff', 'VG']
NUM_IMG=3097 # 2048, 3097, 5096



        cd ${LAYOUT_WORKSPACE}

python -m torch.distributed.launch  \
      --nproc_per_node 2 \
      scripts/classifier-free_sample_for_layout.py  \
      --config_file ./configs/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/${VERSION}.yaml \
      sample.pretrained_model_path=./log/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/${VERSION}/ema_0.9999_${CHECKPOINT}.pt \
      sample.log_root=${SAMPLE_ROOT}/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/${VERSION} \
      sample.timestep_respacing=[${STEPS}] \
      sample.sample_suffix=model${CHECKPOINT}_scale${CLASSIFIER_SCALE}_${SAMPLE_METHOD}  \
      sample.classifier_free_scale=${CLASSIFIER_SCALE} \
      sample.sample_method=${SAMPLE_METHOD} \
      sample.save_cropped_images=True sample.fix_seed=False \
           sample.sample_times=${SAMPLE_TIMES} \
         > log/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/${VERSION}/sample_${NUM_IMG}x${SAMPLE_TIMES}_${CHECKPOINT}_scale${CLASSIFIER_SCALE}_steps${STEPS}_${SAMPLE_METHOD}.txt
#           data.parameters.test.deprecated_stuff_ids_txt=../../guangcongzheng/zju_zgc/LDM_Xianpan/Frido/frido/data/image_ids_layout2i_2048.txt \
#           data.parameters.filter_mode=SG2Im \
    #       data.parameters.test.max_num_samples=${NUM_IMG} \


WORKSPACE='/workspace/mnt/storage/guangcongzheng/zju_zgc/LayoutDiffusion'
cd ${WORKSPACE}


VERSION='LayoutDiffusion-v7_large'
CLASSIFIER_SCALE=1.0
SAMPLE_ROOT='/workspace/mnt/storage/guangcongzheng/zju_zgc_backup/samples'

CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch  \
      --nproc_per_node 2 \
      scripts/classifier-free_sample_for_layout.py  \
      --config_file './configs/COCO-stuff_256x256/LayoutDiffusion_large.yaml' \
       sample.pretrained_model_path='./pretrained_models/COCO-stuff_256x256_LayoutDiffusion_large_ema_1150000' \
       sample.log_root=${SAMPLE_ROOT}/COCO-stuff_256x256/LayoutDiffusion_large \
       sample.timestep_respacing=[25] \
       sample.sample_suffix=model1150000_scale${CLASSIFIER_SCALE}_dpm_solver  \
       sample.classifier_free_scale=${CLASSIFIER_SCALE} \
       sample.sample_method='dpm_solver' sample.sample_times=5 \
       sample.save_cropped_images=True sample.fix_seed=False
       data.parameters.test.batch_size=4 \
#       data.parameters.test.deprecated_stuff_ids_txt=../../guangcongzheng/zju_zgc/LDM_Xianpan/Frido/frido/data/image_ids_layout2i_2048.txt