#!/usr/bin/env bash

python train_resnext.py --dataset_path=/home/autel/data/cifar/cifar-10-batches-py/tfrecords \
--mean_img=/home/autel/data/cifar/cifar-10-batches-py/mean_img.npy \
--mode=train \
--log_dir=/home/autel/PycharmProjects/netzoo/log \
--train_batch_size=1 \
--epoch_num=1 \
--gpu_num=1 \
--ps_hosts=localhost:2000 \
--worker_hosts=localhost:2001 \
--job_name=ps \
--task_index=0