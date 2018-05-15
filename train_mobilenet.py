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
import tensorlayer as tl
import tensorflow.contrib.slim.nets as nets

def main(_):
    """
    Main script for training.
    :return: None.
    """

    # Create feature extractor.
    gconf = conf.loadTrainConf()
    # mobilenet = mobileNet.MobileNet(gconf)

    # Prepaire data
    dataset = dt.DataSet(path=gconf['dataset_path'],
                         batchsize=gconf['batch_size'],
                         class_num=gconf['class_num'])

    img_name_batch, img_batch, sizes_batch, class_id_batch, box_num_batch, labels_batch, bboxes_batch = dataset.getNext()
    labels_batch = tf.one_hot(class_id_batch, gconf['class_num'] + 1)
    labels_batch = labels_batch[:, 1:]
    dataset_itr = dataset._itr

    # Predict labels
    input_imgs = tf.placeholder(tf.float32, [None, gconf['input_size'], gconf['input_size'], 3])
    alexnet, _ = nets.alexnet.alexnet_v2(input_imgs, num_classes=20, is_training=True)
    logits = alexnet

    #  Compute los
    labels = tf.placeholder(tf.float32, [None, gconf['class_num']])
    tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    loss = tf.losses.get_total_loss()
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=gconf['learning_rate'], epsilon=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('train_acc', accuracy)



    summary = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

    model_saver = tf.train.Saver()
    with  open('/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/classes.json') as label_name_file:
        class_dict = json.load(label_name_file)

    step_cnt = 0
    with tf.Session() as sess:
        tb_log_writer = tf.summary.FileWriter(gconf['log_dir'], sess.graph)

        sess.run(init_op)

        for epoch_idx in range(gconf['epoch_num']):
            sess.run(dataset_itr.initializer)

            while True:
                step_cnt = step_cnt + 1
                try:
                    # train
                    imgs_input, labels_input = sess.run([img_batch, labels_batch])

                    # lab_pred, lab_batch = sess.run([labels_pred, class_id_batch], feed_dict={input_imgs:imgs_input,
                    #                                                       labels: labels_input});
                    # print(lab_pred, lab_batch)
                    # exit(0)

                    # for img, class_onehot in zip(imgs_input, labels_input):
                    #     utils.visulizeClass(img, class_onehot, class_dict, hold=True)
                    #     plt.waitforbuttonpress()

                    summary_val, loss_val, train_acc, _ = sess.run([summary, loss, accuracy, train_op], feed_dict={input_imgs:imgs_input,
                                                                          labels: labels_input})

                    # summary_val, loss_val, train_acc = sess.run([summary, loss, accuracy],
                    #                                                feed_dict={input_imgs: imgs_input,
                    #                                                       labels: labels_input})
                    if step_cnt % gconf['log_step'] == 0:
                        tb_log_writer.add_summary(summary_val, step_cnt)
                        print('Step %d, loss: %f, train_acc: %f'%(step_cnt, loss_val, train_acc))
                except tf.errors.OutOfRangeError:
                    # log statistics
                    # break
                    break

            model_saver.save(sess, '/home/autel/PycharmProjects/netzoo/models/model_%3d.ckpt' % epoch_idx)

            if epoch_idx % 4 == 0:
                optimizer = tf.train.AdamOptimizer(learning_rate=gconf['learning_rate'])
if __name__ == '__main__':
    tf.app.run()


