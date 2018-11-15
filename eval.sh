#!/usr/bin/env bash

# Setup main variables
export basedir=$PWD
export research=$basedir/models/research

# Read input parameters
MODELNAME=$1
MODELDATE=$2

PIPELINE_CONFIG=$3
if [ -z "$PIPELINE_CONFIG" ]; then
    export PIPELINE_CONFIG=${MODELNAME}
fi

MODEL_DIR=$gs_bucket/model_dir/${PIPELINE_CONFIG}_${MODELDATE}

echo Model evaluation
echo PIPELINE_CONFIG: ${PIPELINE_CONFIG}
echo MODEL_DIR: ${MODEL_DIR}

cd $research
PIPELINE_CONFIG_PATH=$gs_bucket/config/${PIPELINE_CONFIG}.config
echo starting evaluation job eval_${PIPELINE_CONFIG}_${MODELDATE}_`date +%m_%d_%Y_%H_%M_%S`
gcloud ml-engine jobs submit training eval_${PIPELINE_CONFIG}_${MODELDATE}_`date +%m_%d_%Y_%H_%M_%S` \
    --runtime-version 1.9 \
    --job-dir=$MODEL_DIR \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
    --module-name object_detection.model_main \
    --region $gcp_region \
    --scale-tier BASIC_GPU \
    -- \
    --model_dir=$MODEL_DIR \
    --pipeline_config_path=$PIPELINE_CONFIG_PATH \
    --checkpoint_dir=$MODEL_DIR


cd $basedir