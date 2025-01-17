#!/bin/bash

SRC=en_XX
TGT=fr_XX

SCIPAR_DATA=location-to-plain-text-scipar-corpus
# SCIPAR_DATA $ ls
# dev-scipar.en  dev-scipar.fr  test-scipar.en  test-scipar.fr  train-scipar.en  train-scipar.fr

BPE_DATA=location-to-store-the-BPE-DATA

# the pretrained models
MBART_DIR=location-of-mbart-checkpoint

BPE_MODEL=$MBART_DIR/mbart50.ft.1n/sentence.bpe.model
SRC_DICT=$MBART_DIR/mbart50.ft.1n/dict.en_XX.txt
TGT_DICT=$MBART_DIR/mbart50.ft.1n/dict.fr_XX.txt
echo BPE_DATA=$BPE_DATA

######################### encode to BPE sentencepiece ##########################
echo "encoding scipar dataset with mBART50-one-to-many bpe model..."

################ ENCODE FUNCTIONS ###########
SPM_ENCODE=../../scripts/spm_encode.py

# argument $BPE_MODEL_PATH $input_fpath $out_fpath
function encode_file_sentpiece {
    python $SPM_ENCODE \
        --model  $1 \
        --output_format=piece \
        --inputs $2 --outputs $3
}

for data in train dev test
do
    for lang in en fr
    do
    input_fpath=$SCIPAR_DATA/$data-scipar.$lang
    output_fpath=$BPE_DATA/$data.sentpiece_mbart50_1n.$lang\_XX
    encode_file_sentpiece $input_fpath $output_fpath
    done
done

######################### binarize ###################

DATASET_SCIPAR=location-to-store-the-dataset
mkdir -p $DATASET_SCIPAR

function make_dataset {
    # argument: $train_fpath $dev_fpath, $ test_fpath $data_store_path
    for v in $* 
    do 
        echo $v 
    done

    fairseq-preprocess \
        --source-lang $SRC --target-lang $TGT \
        --srcdict ${SRC_DICT}\
        --tgtdict ${TGT_DICT} \
        --trainpref $1 \
        --validpref $2 \
        --testpref $3 \
        --destdir $4 --thresholdtgt 0 --thresholdsrc 0 \
        --workers 40

}


echo "Binarize the dataset..."

echo "Scipar: $DATASET_SCIPAR"
train_fpath=$BPE_DATA/train.sentpiece_mbart50_1n
dev_fpath=$BPE_DATA/dev.sentpiece_mbart50_1n 
test_fpath=$BPE_DATA/test.sentpiece_mbart50_1n 
make_dataset $train_fpath $dev_fpath $test_fpath $DATASET_SCIPAR


