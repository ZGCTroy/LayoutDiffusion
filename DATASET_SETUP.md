# Setup Dataset
We follow [Layout2Im](https://github.com/zhaobozb/layout2im), [LostGAN](https://github.com/WillSuen/LostGANs), use the download/preprocess scripts in [Sg2Im](https://github.com/google/sg2im/tree/master/scripts).

### COCO-stuff 2017 (The deprecated segmentation challenge) 
```bash
  bash bash/download_coco.sh
```

  <details><summary>COCO 2017 Segmentation challenge split file structure</summary>

   ```
   ├── annotations
   │    └── deprecated-challenge2017
   │         └── train-ids.txt
   │         └── val-ids.txt
   │    └── instances_train2017.json
   │    └── instances_val2017.json
   │    └── stuff_train2017.json
   │    └── stuff_val2017.json
   │    └── ...
   ├── images
   │    └── train2017
   │         └── 000000000872.jpg
   │         └── ... 
   │   └── val2017
   │         └── 000000000321.jpg
   │         └── ... 
   ```

   </details>

### Visual Genome
```bash
  # Run the following script to download and unpack the relevant parts of the Visual Genome dataset. 
  # This will create the directory datasets/vg and will download about 15 GB of data to this directory; after unpacking it will take about 30 GB of disk space.
  bash bash/download_vg.sh
  
  # After downloading the Visual Genome dataset, we need to preprocess it. This will split the data into train / val / test splits, consolidate all scene graphs into HDF5 files, and apply several heuristics to clean the data. In particular we ignore images that are too small, and only consider object and attribute categories that appear some number of times in the training set; we also ignore objects that are too small, and set minimum and maximum values on the number of objects and relationships that appear per image.
  # This will create files train.h5, val.h5, test.h5, and vocab.json in the directory datasets/vg.
  python scripts/preprocess_vg.py
```

   <details><summary>Visual Genome file structure</summary>

   ```
   ├── VG_100K
   │   └── captions_val2017.json
   │   └── ...
   └── objects.json
   └── train.json
   └── ...
   ```

   </details>