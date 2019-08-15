#!/usr/bin/env bash

export DATA_DIR=fever
export OUTPUT_DIR=FEVER-bin
mkdir $DATA_DIR
wget -O $DATA_DIR/test.tsv 'https://storage.googleapis.com/poloma-tpu/fever/test.tsv'
wget  -O $DATA_DIR/train.tsv 'https://storage.googleapis.com/poloma-tpu/fever/train.tsv'