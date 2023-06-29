WORKSPACE='/workspace/mnt/storage/guangcongzheng/zju_zgc/LayoutDiffusion'
cd ${WORKSPACE}


CLASSIFIER_SCALE=1.0
SAMPLE_ROOT='/workspace/mnt/storage/guangcongzheng/zju_zgc_backup/samples'

CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch  \
      --nproc_per_node 2 \
      scripts/classifier-free_sample_for_layout.py  \
      --config_file './configs/COCO-stuff_256x256/latent_LayoutDiffusion_large.yaml' \
       sample.pretrained_model_path='./log/COCO-stuff_256x256/latent_LayoutDiffusion_large/ema_1150000.pt' \
       sample.log_root=${SAMPLE_ROOT}/COCO-stuff_256x256/latent_LayoutDiffusion_large \
       sample.timestep_respacing=[25] \
       sample.sample_suffix=model1150000_scale${CLASSIFIER_SCALE}_dpm_solver  \
       sample.classifier_free_scale=${CLASSIFIER_SCALE} \
       sample.sample_method='dpm_solver' sample.sample_times=1 \
       data.parameters.test.max_num_samples=8 data.parameters.test.batch_size=4 \
       sample.save_images_with_bboxs=True sample.save_sequence_of_obj_imgs=True sample.save_cropped_images=True sample.fix_seed=False