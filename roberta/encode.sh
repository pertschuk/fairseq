#!/usr/bin/env bash

wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'

export OUTPUT_DIR=fever_output
export DATA_DIR=fever

for SPLIT in train dev; do
    for INPUT in 0 1; do
        python -m examples.roberta.multiprocessing_bpe_encoder \
            --encoder-json encoder.json \
            --vocab-bpe vocab.bpe \
            --inputs "$DATA_DIR/$SPLIT.input$INPUT" \
            --outputs "$DATA_DIR/$SPLIT.input$INPUT.bpe" \
            --workers 60 \
            --keep-empty
    done
done

echo 'preprocessing data'

wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

fairseq-preprocess \
    --only-source \
    --trainpref "$DATA_DIR/train.input0.bpe" \
    --validpref "$DATA_DIR/dev.input0.bpe" \
    --destdir "$OUTPUT_DIR/input0" \
    --workers 60 \
    --srcdict dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "$DATA_DIR/train.input1.bpe" \
    --validpref "$DATA_DIR/dev.input1.bpe" \
    --destdir "$OUTPUT_DIR/input1" \
    --workers 60 \
    --srcdict dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "$DATA_DIR/train.label" \
    --validpref "$DATA_DIR/dev.label" \
    --destdir "$OUTPUT_DIR/label" \
    --workers 60