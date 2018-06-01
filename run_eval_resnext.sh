#!/usr/bin/env bash

python train_resnext.py --dataset_path=/home/autel/data/cifar/cifar-10-batches-py/tfrecords/eval \
--mean_img=/home/autel/data/cifar/cifar-10-batches-py/mean_img.npy \
--mode=eval \
--model_dir=/home/autel/PycharmProjects/netzoo/models \
--log_dir=/home/autel/PycharmProjects/netzoo/log \
--batch_size=1