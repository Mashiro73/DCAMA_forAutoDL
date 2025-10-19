#!/bin/bash

echo "========================================================"
echo "Starting Training for Fold 0"
echo "========================================================"

python \
./train.py --datapath "../" \
           --benchmark fssd12 \
           --fold 0 \
           --bsz 100 \
           --nworker 16 \
           --backbone segman \
           --feature_extractor_path "backbones/SegMAN_Encoder_b.pth.tar" \
           --logpath "./logs" \
           --lr 1e-4 \
           --nepoch 500

echo "========================================================"
echo "Finished Training for Fold 0"
echo "========================================================"