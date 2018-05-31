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
    'gpu_num': 'number of gpu device'
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

    'class_num',
    'learning_rate',
    'weight_decay',
    'momentum',
    'input_size',
    'card',

    'is_train',
    'gpu_num'
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

    return flags.FLAGS


def checkInputParam(FLAGS):
    if FLAGS.mode is 'train' and FLAGS.dataset_path is None:
        raise RuntimeError('You must specify --training_file_pattern for training.')

    if FLAGS.log_dir is None:
        raise RuntimeError('You must specify --log_dir for training.')

    if FLAGS.mean_img is None:
        raise RuntimeError('You must specify --mean_img for training.')


def initParam(input_flag):
    params = Param(
        dataset_path=input_flag.dataset_path,
        mean_img=input_flag.mean_img,
        mode=input_flag.mode,

        log_step=input_flag.log_step,
        log_dir=input_flag.log_dir,

        train_batch_size=input_flag.train_batch_size,
        epoch_num=input_flag.epoch_num,

        class_num=10,
        learning_rate=0.1,
        weight_decay=5e-4,
        momentum=0.9,
        input_size=32,
        card=10,
        is_train=(input_flag.mode == 'train'),
        gpu_num=input_flag.gpu_num
    )

    return params


FLAGS = inputParam()


def main(_):
    """
    Main script for training.
    :return: None.
    """

    gconf = initParam(FLAGS)

    with tf.device('/cpu:0'):
        # Prepaire data
        dataset = CifarDataSet(path=gconf.dataset_path,
                               batchsize=gconf.train_batch_size,
                               class_num=gconf.class_num,
                               mean_img=gconf.mean_img)

        dataset_itr = dataset._itr

        # create train op
        learning_rate = tf.placeholder(tf.float32, name='LR')
        optimizer = tf.train.MomentumOptimizer(learning_rate, gconf.momentum)

        # multi-GPU support
        tower_grads = []
        accuracys = []
        for i in range(gconf.gpu_num):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('gputower_%d' % i) as scope:
                    img_name_batch, img_batch, size_batch, class_id_batch, label_name_batch = dataset.getNext()
                    labels_onehot = tf.one_hot(class_id_batch, gconf.class_num)

                    # Predict labels
                    # img_batch = tf.cast(input_imgs, tf.float32)
                    img_batch = tf.cast(img_batch, dtype=tf.float32)
                    resnext = ResNeXt29(conf=gconf, input=img_batch)
                    logits = resnext.get_output()

                    #  Compute loss
                    tf.losses.softmax_cross_entropy(onehot_labels=labels_onehot, logits=logits, scope=scope)
                    vars_train = tf.trainable_variables(scope='ResNeXt29')
                    weight_loss = gconf.weight_decay * tf.add_n(
                        [tf.nn.l2_loss(v) for v in vars_train if 'bias' not in v.name])
                    tf.add_to_collection('losses', weight_loss)
                    losses = tf.get_collection('losses', scope=scope)
                    loss = tf.add_n(losses, name='total_loss')

                    # create train accuracy
                    correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels_onehot, axis=1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    accuracys.append(accuracy)

                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)

        # average grads through all GPUs
        avg_grads = utils.avg_grads(tower_grads)

        # apply gradients
        apply_grad_op = optimizer.apply_gradients(avg_grads, global_step=tf.train.get_global_step())

        total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('loss', total_loss)

        # compute mean accuracy
        mean_acc = tf.reduce_mean(tf.add_n(accuracys))
        tf.summary.scalar('train_acc', mean_acc)

        summary = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        model_saver = tf.train.Saver()

        step_cnt = 0
        total_duration = 0
        with tf.Session() as sess:
            tb_log_writer = tf.summary.FileWriter(gconf.log_dir, sess.graph)

            sess.run(init_op)

            # Find last checkpoint
            ckpt_name, epoch_start = utils.findLastCkpt(path.join(home, 'models'))
            if ckpt_name != '':
                model_saver.restore(sess, path.join(home, 'models', ckpt_name))

            lr = gconf.learning_rate
            for epoch in range(epoch_start, gconf.epoch_num):
                sess.run(dataset_itr.initializer)

                if epoch == 150:
                    lr = gconf.learning_rate / 10.0

                if epoch == 225:
                    lr = gconf.learning_rate / 100.0

                while True:
                    start_time = time.time()
                    step_cnt = step_cnt + 1
                    try:
                        # for i in range(gconf['batch_size']):
                        #     # vis imgs and labels
                        #     utils.visulizeClassByName(imgs_input[i], label_name_input[i], hold=True)
                        #     plt.waitforbuttonpress()

                        summary_val, loss_val, mean_acc_val, _ = sess.run([summary, total_loss, mean_acc, apply_grad_op],
                                                                       feed_dict={learning_rate: lr})

                        if step_cnt % gconf.log_step == 0:
                            duration = time.time() - start_time
                            total_duration += duration
                            tb_log_writer.add_summary(summary_val, step_cnt)
                            print('time: %s, Step %d, loss: %f, train_acc: %.4f, duration: %.3fs, total_duration: %.3fs' \
                                  % (datetime.datetime.now(), step_cnt, loss_val, mean_acc_val, duration, total_duration))

                    except tf.errors.OutOfRangeError:
                        break

                if epoch % 10 == 0:
                    model_saver.save(sess, 'models/model_%03d.ckpt' % epoch)


if __name__ == '__main__':
    tf.app.run()
