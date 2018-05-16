# -*- coding: utf-8 -*-

"""
Scripts for training a ResNeXt network.
"""

import tensorflow as tf
import Datasets.PascalDataset as dt
import common.config as conf
import json
from network.ResNeXt29 import ResNeXt29

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

    img_name_batch, img_batch, sizes_batch, class_id_batch = dataset.getNext()
    labels_batch = tf.one_hot(class_id_batch, gconf['class_num'])
    dataset_itr = dataset._itr

    # Predict labels
    # img_batch = tf.cast(input_imgs, tf.float32)
    input_imgs = tf.placeholder(tf.float32, [None, gconf['input_size'], gconf['input_size'], 3])
    resnet, _ =  ResNeXt29(conf=gconf, input=input_imgs)
    logits = resnet.outputs

    #  Compute loss
    labels = tf.placeholder(tf.float32, [None, gconf['class_num']])
    tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    loss = tf.losses.get_total_loss()
    tf.summary.scalar('loss', loss)

    learning_rate = tf.placeholder(tf.float32, name='LR')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('train_acc', accuracy)

    summary = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

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


                    summary_val, loss_val, train_acc, _ = sess.run([summary, loss, accuracy, train_op],
                                                                   feed_dict={input_imgs:imgs_input,
                                                                              labels: labels_input,
                                                                              learning_rate:gconf['learning_rate']})

                    if step_cnt % gconf['log_step'] == 0:
                        tb_log_writer.add_summary(summary_val, step_cnt)
                        print('Step %d, loss: %f, train_acc: %f'%(step_cnt, loss_val, train_acc))
                except tf.errors.OutOfRangeError:
                    # log statistics
                    # break
                    break

if __name__ == '__main__':
    tf.app.run()