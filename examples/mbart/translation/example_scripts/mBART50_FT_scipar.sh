#!/bin/bash
###### 1 GPU RTX-A6000 with 12 CPU with --mem-per-cpu=8G ######
###### Pytorch 1.13.1, transformers 4.45.23 ######


SRC=en_XX
TGT=fr_XX

archi=mbart_large 

# location of the mbart checkpoint mbart50.ft.1n
MBART_DIR==location-of-mbart-checkpoint 
lang_pairs="en_XX-fr_XX"
lang_list=$MBART_DIR/ML50_langs.txt
PRETRAIN=$MBART_DIR/model.pt
BPE_MODEL=$MBART_DIR/sentence.bpe.model


DATASET_SCIPAR=location-to-the-dataset
SAVE_DIR=location-to-store-the-checkpoints
mkdir -p $SAVE_DIR

fairseq-train \
    $DATASET_SCIPAR \
    --arch $archi \
    --finetune-from-model $PRETRAIN \
    --encoder-normalize-before --decoder-normalize-before \
    --layernorm-embedding \
    --task translation_multi_simple_epoch \
    --sampling-method "temperature" \
    --sampling-temperature 1.5 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr 3e-05 --warmup-updates 2500  \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
    --max-tokens 4096 \
    --update-freq 4 \
    --save-interval 1 --save-interval-updates 40000 --keep-interval-updates 5 \
    --no-epoch-checkpoints \
    --seed 222 \
    --log-format simple --log-interval 100 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.5, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe sentencepiece\
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 5 \
    --num-workers 0 \
    --amp \
    --ddp-backend legacy_ddp \
    --save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR
