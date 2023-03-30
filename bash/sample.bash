
LAYOUT_WORKSPACE='/workspace/mnt/storage/guangcongzheng/zju_zgc/LayoutDiffusion'
cd ${LAYOUT_WORKSPACE}
nohup df -h > df_h.txt
conda activate

pip install omegaconf
pip install -e ../dpm_solver

#rm -rf /opt/conda/lib/python3.8/site-packages/layout_diffusion
#rm -rf ./layout_diffusion.egg-info
#
python setup.py build develop

cp /workspace/mnt/storage/guangcongzheng/zju_zgc/TTUR/classify_image_graph_def.pb /tmp/classify_image_graph_def.pb

WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=$((RANDOM + 10000))


VERSION='LayoutDiffusion-v7_large'
SAMPLE_METHOD='dpm_solver' # ['dpm_solver', 'ddpm', 'ddim']
SAMPLE_TIMES=5
#STEPS=10
#SAMPLE_ROOT='/workspace/mnt/storage/3150104097/zju_tiny_backup/samples'
SAMPLE_ROOT='/workspace/mnt/storage/guangcongzheng/zju_zgc_backup/samples'

IMAGE_SIZE=128
DATASET='COCO-stuff' # ['COCO-stuff', 'VG']
NUM_IMG=3097 # 2048, 3097, 5096


for STEPS in 25
  do
  for CLASSIFIER_SCALE in 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2
    do
    for CHECKPOINT in 0300000
      do
        cd ${LAYOUT_WORKSPACE}

      python -m torch.distributed.launch  \
          --nnodes $WORLD_SIZE \
          --node_rank $RANK \
          --nproc_per_node 8 \
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
           sample.save_cropped_images=True sample.fix_seed=False \
           sample.sample_times=${SAMPLE_TIMES} \
         > log/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/${VERSION}/sample_${NUM_IMG}x${SAMPLE_TIMES}_${CHECKPOINT}_scale${CLASSIFIER_SCALE}_steps${STEPS}_${SAMPLE_METHOD}.txt
#           data.parameters.test.deprecated_stuff_ids_txt=../../guangcongzheng/zju_zgc/LDM_Xianpan/Frido/frido/data/image_ids_layout2i_2048.txt \
#           data.parameters.filter_mode=SG2Im \
    #       data.parameters.test.max_num_samples=${NUM_IMG} \

        cd /workspace/mnt/storage/guangcongzheng/zju_zgc/TTUR

        GENERATED_IMGS_DIR=${SAMPLE_ROOT}/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/${VERSION}/conditional_[${STEPS}]/sample${NUM_IMG}x${SAMPLE_TIMES}/model${CHECKPOINT}_scale${CLASSIFIER_SCALE}_${SAMPLE_METHOD}/generated_imgs
        REAL_IMGS_DIR=${SAMPLE_ROOT}/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/${VERSION}/conditional_[${STEPS}]/sample${NUM_IMG}x${SAMPLE_TIMES}/model${CHECKPOINT}_scale${CLASSIFIER_SCALE}_${SAMPLE_METHOD}/real_imgs

        nohup ./TTUR_v2/fid.py --gpu 0,1 \
            ${GENERATED_IMGS_DIR} \
            ${REAL_IMGS_DIR} \
            > ${LAYOUT_WORKSPACE}/log/${DATASET}_${IMAGE_SIZE}x${IMAGE_SIZE}/${VERSION}/evaluate_${NUM_IMG}x${SAMPLE_TIMES}_${CHECKPOINT}_scale${CLASSIFIER_SCALE}_steps${STEPS}_${SAMPLE_METHOD}.txt

        done
    done
done