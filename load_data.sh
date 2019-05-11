#!/usr/bin/env bash
set -e
export DATASET_FOLDER="celeba"

kaggle datasets download -d jessicali9530/celeba-dataset
unzip celeba-dataset.zip -d ${DATASET_FOLDER}
unzip ${DATASET_FOLDER}/img_align_celeba.zip -d ${DATASET_FOLDER}/
rm ${DATASET_FOLDER}/img_align_celeba.zip