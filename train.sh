#!/usr/bin/env bash

# Setup main variables
export basedir=$PWD
export research=$basedir/models/research

# Read input parameters
MODE=$1
MODELNAME=$2
MODELDATE=$3

PIPELINE_CONFIG=$4
if [ -z "$PIPELINE_CONFIG" ]; then
    export PIPELINE_CONFIG=${MODELNAME}
fi

CHECKPOINT_NAME=${MODELNAME}_${MODELDATE}
CHECKPOINT_DIR=$gs_bucket/checkpoints/${CHECKPOINT_NAME}
MODEL_DIR=$gs_bucket/model_dir/${PIPELINE_CONFIG}_${MODELDATE}
YAML_CONFIG=$basedir/config/cloud.yml

echo PIPELINE_CONFIG: ${PIPELINE_CONFIG}
echo CHECKPOINT_DIR: ${CHECKPOINT_DIR}
echo MODEL_DIR: ${MODEL_DIR}

# 1. Transfer pretreined model to Google Storage
gsutil -q stat ${CHECKPOINT_DIR}/model.ckpt.index

return_value=$?

cd $basedir
if [ $return_value != 0 ]; then
    if [ ! -d checkpoints/${CHECKPOINT_NAME} ]; then
        echo Downloading pretrained model
        wget http://storage.googleapis.com/download.tensorflow.org/models/object_detection/${CHECKPOINT_NAME}.tar.gz -P checkpoints/
        tar -xvf checkpoints/${CHECKPOINT_NAME}.tar.gz -C checkpoints/
        rm checkpoints/${CHECKPOINT_NAME}.tar.gz
    fi

    gsutil cp checkpoints/${CHECKPOINT_NAME}/model.ckpt.* ${CHECKPOINT_DIR}/

    echo Pretrained model has been transfered to Google Storage.
    #rm -r $basedir/checkpoints/${CHECKPOINT_NAME}
fi

# 2. Update config with checkpoint name and copy to Google Storage
cp config/${PIPELINE_CONFIG}.config config/${PIPELINE_CONFIG}.bcp.config
#echo sed -i "s|fine_tune_checkpoint:.+?\n|fine_tune_checkpoint: "${CHECKPOINT_DIR}\model.ckpt"\n|g" \
perl -p -i -e "s|fine_tune_checkpoint:.+?\n|fine_tune_checkpoint: \"${CHECKPOINT_DIR}/model.ckpt\"\n|g" config/${PIPELINE_CONFIG}.config

gsutil cp config/${PIPELINE_CONFIG}.config $gs_bucket/config/

# 3. Run job
PIPELINE_CONFIG_PATH=$gs_bucket/config/${PIPELINE_CONFIG}.config
cd $research

if [ "$MODE" == "gpu" ]; then
    echo Starting training job train_${PIPELINE_CONFIG}_${MODELDATE}_gpu_`date +%m_%d_%Y_%H_%M_%S`
    gcloud ml-engine jobs submit training train_${PIPELINE_CONFIG}_${MODELDATE}_gpu_`date +%m_%d_%Y_%H_%M_%S` \
        --runtime-version 1.9 \
        --job-dir=$MODEL_DIR \
        --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
        --module-name object_detection.model_main \
        --region $gcp_region \
        --config $YAML_CONFIG \
        -- \
        --model_dir=$MODEL_DIR \
        --pipeline_config_path=$PIPELINE_CONFIG_PATH
elif [ "$MODE" == "tpu" ]; then
    echo Starting training job train_${PIPELINE_CONFIG}_${MODELDATE}_tpu_`date +%m_%d_%Y_%H_%M_%S`
    gcloud ml-engine jobs submit training train_${PIPELINE_CONFIG}_${MODELDATE}_tpu_`date +%m_%d_%Y_%H_%M_%S` \
    --job-dir=$MODEL_DIR \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
    --module-name object_detection.model_tpu_main \
    --runtime-version 1.9 \
    --scale-tier BASIC_TPU \
    --region $gcp_region \
    -- \
    --tpu_zone $gcp_region \
    --model_dir=$MODEL_DIR \
    --pipeline_config_path=$PIPELINE_CONFIG_PATH
else
    echo $MODE is unnown mode!
fi

cd $basedir