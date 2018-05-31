# -*- coding: utf-8 -*-

"""
Scripts for training a ResNeXt network.
"""

import tensorflow as tf
import common.config as conf
from network.ResNeXt29 import ResNeXt29
import common.utils as utils
from Datasets.CifarDataset import CifarDataSet
from os import path
from tensorflow.python.framework import ops
from collections import namedtuple
from absl import flags
import time
import datetime

home = path.dirname(path.realpath(__file__))

help_dict = {
    'dataset_path': 'directory contains tfrecords',
    'mean_img': 'path to mean image file',
    'mode': 'train or eval',

    'log_step': 'log each log_step',
    'log_dir': 'log directory',

    'train_batch_size': 'batch size for training',
    'epoch_num': 'epoch number',
    'gpu_num': 'number of gpu',
    'exam_per_epoch': 'examples per epoch',

    # multi-machine training support
    'ps_hosts': 'parameter hosts, seperated by comma',
    'worker_hosts': 'worker hosts, seperated by comma',
    'job_name': 'ps or worker',
    'task_index': 'task index starting from zero'
}


def arg_def(name, default_val):
    return name, default_val, help_dict[name]


Param = namedtuple('ParamStruct', [
    'dataset_path',
    'mean_img',
    'mode',

    'log_step',
    'log_dir',

    'train_batch_size',
    'epoch_num',
    'exam_per_epoch',

    'class_num',
    'learning_rate',
    'weight_decay',
    'momentum',
    'input_size',
    'card',

    'is_train',
    'gpu_num',

    'ps_hosts',
    'worker_hosts',
    'job_name',
    'task_index'
])


def inputParam():
    flags.DEFINE_string(*arg_def('dataset_path', ''))
    flags.DEFINE_string(*arg_def('mean_img', ''))
    flags.DEFINE_string(*arg_def('mode', 'train'))

    flags.DEFINE_integer(*arg_def('log_step', 10))
    flags.DEFINE_string(*arg_def('log_dir', ''))

    flags.DEFINE_integer(*arg_def('train_batch_size', 32))
    flags.DEFINE_integer(*arg_def('epoch_num', 300))
    flags.DEFINE_integer(*arg_def('gpu_num', 1))
    flags.DEFINE_integer(*arg_def('exam_per_epoch', 50000))

    flags.DEFINE_string(*arg_def('ps_hosts', ''))
    flags.DEFINE_string(*arg_def('worker_hosts', ''))
    flags.DEFINE_string(*arg_def('job_name', ''))
    flags.DEFINE_integer(*arg_def('task_index', 0))

    return flags.FLAGS


def checkInputParam(FLAGS):
    if FLAGS.mode is 'train' and FLAGS.dataset_path is None:
        raise RuntimeError('You must specify --training_file_pattern for training.')

    if FLAGS.log_dir is None:
        raise RuntimeError('You must specify --log_dir for training.')

    if FLAGS.mean_img is None:
        raise RuntimeError('You must specify --mean_img for training.')

    if FLAGS.ps_hosts is None:
        raise RuntimeError('You must specify --ps_hosts for training.')

    if FLAGS.worker_hosts is None:
        raise RuntimeError('You must specify --worker_hosts for training.')

    if FLAGS.job_name is None:
        raise RuntimeError('You must specify --job_name for training.')

    if FLAGS.task_index is None:
        raise RuntimeError('You must specify --task_index for training.')

def initParam(input_flag):
    params = Param(
        dataset_path=input_flag.dataset_path,
        mean_img=input_flag.mean_img,
        mode=input_flag.mode,

        log_step=input_flag.log_step,
        log_dir=input_flag.log_dir,

        train_batch_size=input_flag.train_batch_size,
        epoch_num=input_flag.epoch_num,
        exam_per_epoch=input_flag.exam_per_epoch,

        class_num=10,
        learning_rate=0.1,
        weight_decay=5e-4,
        momentum=0.9,
        input_size=32,
        card=8,
        is_train=(input_flag.mode == 'train'),
        gpu_num=input_flag.gpu_num,

        ps_hosts=input_flag.ps_hosts,
        worker_hosts=input_flag.worker_hosts,
        job_name=input_flag.job_name,
        task_index=input_flag.task_index
    )

    return params


FLAGS = inputParam()

def lr_schedule(base_lr, global_step):
    lr = base_lr
    lr = tf.where(global_step > 150, base_lr/10, base_lr)
    lr = tf.where(global_step > 225, lr / 10, lr)

    return lr

def main(_):
    """
    Main script for training.
    :return: None.
    """

    gconf = initParam(FLAGS)

    # parse ps hosts and worker hosts
    ps_hosts = gconf.ps_hosts.split(',')
    wk_hosts = gconf.worker_hosts.split(',')

    # create cluster specific
    clusterSpec = tf.train.ClusterSpec({
        'ps': ps_hosts,
        'worker': wk_hosts
    })

    # create server
    server = tf.train.Server(clusterSpec, job_name=gconf.job_name, task_index=gconf.task_index)

    if gconf.job_name == 'ps':
        # parameter server start and block
        server.join()
    elif gconf.job_name == 'worker':
        # the machine execute task_0 is regarded as chief node.
        is_chief = gconf.task_index == 0

        # worker server do regular training
        with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:%d' % gconf.task_index,
                                                      cluster=clusterSpec)):

            # Prepaire data
            dataset = CifarDataSet(path=gconf.dataset_path,
                                   batchsize=gconf.train_batch_size,
                                   class_num=gconf.class_num,
                                   mean_img=gconf.mean_img)

            img_name_batch, img_batch, size_batch, class_id_batch, label_name_batch = dataset.getNext()
            labels_onehot = tf.one_hot(class_id_batch, gconf.class_num)

            # Predict labels
            resnext =  ResNeXt29(conf=gconf, input=img_batch)
            logits = resnext.get_output()

            #  Compute loss
            tf.losses.softmax_cross_entropy(onehot_labels=labels_onehot, logits=logits)

            vars_train = tf.trainable_variables(scope='ResNeXt29')
            weight_loss = gconf.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in vars_train if 'bias' not in v.name])
            tf.losses.add_loss(weight_loss, loss_collection=ops.GraphKeys.REGULARIZATION_LOSSES)

            loss = tf.losses.get_total_loss()
            tf.summary.scalar('loss', loss)

            # create train op
            learning_rate = tf.placeholder(tf.float32, name='LR')
            global_step = tf.train.get_or_create_global_step()
            lr = lr_schedule(learning_rate, global_step)

            optimizer = tf.train.MomentumOptimizer(learning_rate, gconf.momentum)
            train_op = optimizer.minimize(loss, global_step=global_step)

            # create train accuracy
            correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels_onehot, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('train_acc', accuracy)

        summary = tf.summary.merge_all()

        # create stop conditon hooks
        hooks = [tf.train.StopAtStepHook(gconf.epoch_num * gconf.exam_per_epoch)]

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               checkpoint_dir=gconf.log_dir,
                                               hooks=hooks,
                                               log_step_count_steps=1) as mon_sess:
                if is_chief:
                    mon_sess.run(dataset._itr.initializer)
                while not mon_sess.should_stop():
                        # train
                        mon_sess.run([train_op], feed_dict={learning_rate:gconf.learning_rate})

if __name__ == '__main__':
    tf.app.run()