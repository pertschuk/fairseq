#!/usr/bin/env bash

export TOTAL_NUM_UPDATES=7812  # 10 epochs through IMDB for bsz 32

export WARMUP_UPDATES=7432      # 6 percent of the number of updates
export LR=1e-05                # Peak LR for polynomial LR scheduler.
export NUM_CLASSES=2
export MAX_SENTENCES=4        # Batch size.
export ROBERTA_PATH=roberta.large/model.pt

CUDA_VISIBLE_DEVICES=0 python train.py airbnb_train/ \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --truncate-sequence \
    --find-unused-parameters