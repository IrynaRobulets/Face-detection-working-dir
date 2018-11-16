#!/usr/bin/env bash

# NOTE! DO NOT USE IT AS A BATCH SCRIPT!
# THIS FILE CONTAINS ONLY NOTES OF EXAMPLE OF RESEARCH

set -e

# 1. Setup env variables
export basedir=$PWD
export research=$PWD/models/research

export PROJECT=diplomawork-166315
export gcp_region=europe-west1
export gs_bucket=gs://dubnevych-workspace

echo $basedir
echo $research
echo $gcp_region
echo $gs_bucket

gcloud auth application-default login

# 1. Download and convert Wider Face dataset to TF format

widerface_init_data.sh


# 2. Run training with evaluation and copy download results (ONE BY ONE)

train.sh gpu ssd_mobilenet_v1_coco 2018_01_28
gsutil cp -r $gs_bucket/model_dir/ssd_mobilenet_v1_coco_2018_01_28 trained_models/

train.sh gpu ssd_mobilenet_v1_coco 2018_01_28 ssd_mobilenet_v1_focal_loss_coco
gsutil cp -r $gs_bucket/model_dir/ssd_mobilenet_v1_focal_loss_coco_2018_01_28 trained_models/

train.sh gpu ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync 2018_07_03
gsutil cp -r $gs_bucket/model_dir/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03 trained_models/

train.sh gpu ssd_mobilenet_v2_coco 2018_03_29
gsutil cp -r $gs_bucket/model_dir/ssd_mobilenet_v2_coco_2018_03_29 trained_models/

train.sh gpu ssdlite_mobilenet_v2_coco 2018_05_09
gsutil cp -r $gs_bucket/model_dir/ssdlite_mobilenet_v2_coco_2018_05_09 trained_models/

# 3. Cleanup GCP bucket
gsutil rm -r $gs_bucket/*.*

# 4. Collecting data in tensorboard (ONE BY ONE)
# To see result open localhost:6006 in your browser

tensorboard --logdir=trained_models/ssd_mobilenet_v1_focal_loss_coco_2018_01_28
tensorboard --logdir=trained_models/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03
tensorboard --logdir=trained_models/ssdlite_mobilenet_v2_coco_2018_05_09
tensorboard --logdir=trained_models/ssd_mobilenet_v2_coco_2018_03_29


# *********** GRANT TPU ACCOUNT WITH PERMISSIONS

curl -H "Authorization: Bearer $(gcloud auth print-access-token)"  https://ml.googleapis.com/v1/projects/${PROJECT}:getConfig

# You should receive something like this. Then take the value of "tpuServiceAccount" property.
# That is your TPU service account name.
#{
#  "serviceAccount": "service-1060824920780@cloud-ml.google.com.iam.gserviceaccount.com",
#  "serviceAccountProject": "906872764684",
#  "config": {
#    "tpuServiceAccount": "service-906872764684@cloud-tpu.iam.gserviceaccount.com"
#  }
#}

# Grant it with access to your project
export TPU_ACCOUNT=service-906872764684@cloud-tpu.iam.gserviceaccount.com
gcloud projects add-iam-policy-binding $PROJECT --member serviceAccount:$TPU_ACCOUNT --role roles/ml.serviceAgent

#gsutil acl ch -u service-906872764684@cloud-tpu.iam.gserviceaccount.com:READER $gs_bucket
#gsutil acl ch -u service-906872764684@cloud-tpu.iam.gserviceaccount.com:WRITER $gs_bucket