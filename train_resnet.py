# -*- coding: utf-8 -*-

"""
Scripts for training a ResNeXt network.
"""

import tensorflow as tf
import common.config as conf
import json
from network.ResNeXt29 import ResNeXt29
import common.utils as utils
from Datasets.CifarDataset import CifarDataSet
import matplotlib.pyplot as plt
from os import path

home = path.dirname(path.realpath(__file__))
def main(_):
    """
    Main script for training.
    :return: None.
    """

    # Create feature extractor.
    gconf = conf.loadTrainConf()
    # mobilenet = mobileNet.MobileNet(gconf)

    # Prepaire data
    dataset = CifarDataSet(path=gconf['dataset_path'],
                         batchsize=gconf['batch_size'],
                         class_num=gconf['class_num'],
                         mean_img = gconf['mean_img'])

    img_name_batch, img_batch, size_batch, class_id_batch, label_name_batch = dataset.getNext()
    labels_onehot = tf.one_hot(class_id_batch, gconf['class_num'])
    dataset_itr = dataset._itr

    # Predict labels
    # img_batch = tf.cast(input_imgs, tf.float32)
    input_imgs = tf.placeholder(tf.float32, [None, gconf['input_size'], gconf['input_size'], 3])
    resnet =  ResNeXt29(conf=gconf, input=input_imgs)
    logits = resnet.get_output()

    #  Compute loss
    labels = tf.placeholder(tf.float32, [None, gconf['class_num']])
    tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    loss = tf.losses.get_total_loss()
    tf.summary.scalar('loss', loss)

    # create train op
    learning_rate = tf.placeholder(tf.float32, name='LR')
    optimizer = tf.train.MomentumOptimizer(learning_rate, gconf['momentum'])
    train_op = optimizer.minimize(loss)

    # create train accuracy
    correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('train_acc', accuracy)

    summary = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()
    model_saver = tf.train.Saver()

    step_cnt = 0
    with tf.Session() as sess:
        tb_log_writer = tf.summary.FileWriter(gconf['log_dir'], sess.graph)

        sess.run(init_op)

        # Find last checkpoint
        ckpt_name, epoch_start = utils.findLastCkpt(path.join(home,'models'))
        if ckpt_name != '':
            model_saver.restore(sess, path.join(home, 'models', ckpt_name))

        lr = gconf['learning_rate']
        for epoch in range(epoch_start, gconf['epoch_num']):
            sess.run(dataset_itr.initializer)

            if epoch == 150:
                lr = gconf['learning_rate'] / 10.0

            if epoch == 225:
                lr = gconf['learning_rate'] / 100.0

            while True:
                step_cnt = step_cnt + 1
                try:
                    # train
                    imgs_input, labels_input, label_name_input = sess.run([img_batch, labels_onehot, label_name_batch])


                    # for i in range(gconf['batch_size']):
                    #     # vis imgs and labels
                    #     utils.visulizeClassByName(imgs_input[i], label_name_input[i], hold=True)
                    #     plt.waitforbuttonpress()

                    
                    summary_val, loss_val, train_acc, _ = sess.run([summary, loss, accuracy, train_op],
                                                                   feed_dict={input_imgs:imgs_input,
                                                                              labels: labels_input,
                                                                              learning_rate:lr})

                    if step_cnt % gconf['log_step'] == 0:
                        tb_log_writer.add_summary(summary_val, step_cnt)
                        print('Step %d, loss: %f, train_acc: %f'%(step_cnt, loss_val, train_acc))

                except tf.errors.OutOfRangeError:
                    break

            if epoch % 10 == 0:
                model_saver.save(sess, 'models/model_%03d.ckpt' % epoch)

if __name__ == '__main__':
    tf.app.run()