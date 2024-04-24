#!/bin/bash
source /home/xiaobin/anaconda3/bin/activate nlp_project
python -m training.main \
    --batch-size 64 \
    --precision amp \
    --workers 1 \
    --report-to tensorboard \
    --save-frequency 1 \
    --logs "training/logs" \
    --dataset-type csv \
    --csv-separator "," \
    --train-data "../../coco/annotations/train_neg_clip_merged_decompos_v2.csv" \
    --csv-img-key filepath \
    --csv-caption-key title \
    --csv-num-cap-key num_pos_caption \
    --warmup 50 \
    --lr=5e-6 \
    --wd=0.1 \
    --epochs=5 \
    --model RN50 \
    --pretrained openai