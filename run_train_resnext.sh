#!/usr/bin/env bash

python train_resnext.py --dataset_path=/home/ubuntu/data/cifar-10-batches-py/tfrecords \
--mean_img=/home/ubuntu/data/cifar-10-batches-py/mean_img.npy \
--mode=train \
--log_dir=/home/ubuntu/models/ \
--train_batch_size=32 \
--epoch_num=300 \
--gpu_num=1 2>&1 | tee /home/ubuntu/models/train_log.txt