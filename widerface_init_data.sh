#!/usr/bin/env bash

python3.exe download_widerface.py
python scripts/create_widerface_tf_record.py \
  --input_dir=downloads \
  --output_dir=data

gsutil cp $basedir/data/wider_train.tfrecord $gs_bucket/data/
gsutil cp $basedir/data/wider_val.tfrecord $gs_bucket/data/
gsutil cp $basedir/data/label_map.pbtxt $gs_bucket/data/

rm -r downloads/WIDER_train
rm -r downloads/WIDER_val