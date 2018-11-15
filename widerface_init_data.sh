#!/usr/bin/env bash

python3.exe download_widerface.py
python3.exe create_widerface_tf_record.py

rm -r downloads/WIDER_train
rm -r downloads/WIDER_val