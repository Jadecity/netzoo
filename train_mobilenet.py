# -*- coding: utf-8 -*-

"""
Scripts for training a mobile network.
"""

import tensorflow as tf
import network.ssdConf as ssdConf
import common.dataset as dt
import network.ssdNet as ssdNet
import network.mobileNet as mobileNet
import common.utils as utils
import common.config as conf
import numpy as np
import json
import matplotlib.pyplot as plt

def main(_):
    """
    Main script for training.
    :return: None.
    """

    # Create feature extractor.
    gconf = conf.loadTrainConf()
    mobilenet = mobileNet.MobileNet(gconf)

    # Prepaire data
    dataset = dt.DataSet(path=gconf['dataset_path'],
                         batchsize=gconf['batch_size'],
                         class_num=gconf['class_num'])

    img_name_batch, img_batch, sizes_batch, class_id_batch, box_num_batch, labels_batch, bboxes_batch = dataset.getNext()
    labels_batch = tf.one_hot(class_id_batch, gconf['class_num'] + 1)
    labels_batch = labels_batch[:, 1:]
    dataset_itr = dataset._itr

    # Predict labels
    # img_batch = tf.cast(input_imgs, tf.float32)
    input_imgs = tf.placeholder(tf.float32, [None, gconf['input_size'], gconf['input_size'], 3])
    net_end, _ = mobilenet.predict(input_imgs)
    logits = net_end.outputs

    #  Compute loss
    labels = tf.placeholder(tf.float32, [None, gconf['class_num']])
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
    loss = tf.reduce_mean(loss)
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=gconf['learning_rate'])
    train_op = optimizer.minimize(loss)

    labels_pred = tf.argmax(logits)
    labels_true = tf.argmax(labels_batch)
    correct_prediction = tf.equal(labels_pred, labels_true)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('train_acc', accuracy)

    summary = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()


    with  open('/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/classes.json') as label_name_file:
        class_dict = json.load(label_name_file)

    step_cnt = 0
    with tf.Session() as sess:
        tb_log_writer = tf.summary.FileWriter(gconf['log_dir'], sess.graph)

        sess.run(init_op)

        for _ in range(gconf['epoch_num']):
            sess.run(dataset_itr.initializer)

            while True:
                step_cnt = step_cnt + 1
                try:
                    # train
                    imgs_input, labels_input = sess.run([img_batch, labels_batch])
                    summary_val, loss_val, test_acc, _ = sess.run([summary, loss, accuracy, train_op], feed_dict={input_imgs:imgs_input,
                                                                          labels: labels_input})
                    # for img, class_id in zip(imgs_input, labels_input):
                    #     utils.visulizeClass(img, class_id, class_dict)
                    #     plt.waitforbuttonpress()

                    if step_cnt % gconf['log_step'] == 0:
                        tb_log_writer.add_summary(summary_val, step_cnt)
                        print('Step %d, loss: %f, train_acc: %f'%(step_cnt, loss_val, test_acc))
                except tf.errors.OutOfRangeError:
                    # log statistics
                    # break
                    break

if __name__ == '__main__':
    tf.app.run()


