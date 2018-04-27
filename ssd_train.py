"""
Scripts for training a ssd network.
"""

import tensorflow as tf
import tensorlayer as tl
import network.ssdConf as ssdConf
import network.mobileConf as mobileConf
import common.dataset as dt
import network.ssdNet as ssdNet
import network.mobileNet as mobileNet
import common.utils as utils
import common.config as script_conf



def main(_):
    """
    Main script for training.
    :return: None.
    """

    # Create feature extractor.
    mobile_conf = mobileConf.loadMobileConf()
    ft_extractor = mobileNet.MobileNet(mobile_conf)

    # Load SSD config.
    ssd_conf = ssdConf()
    ssd_net = ssdNet.SSDNet(ssd_conf, ft_extractor)

    # Predict labels and bouding boxes.
    input_img = tf.placeholder(tf.int8, [None, ssd_conf['input_h'], ssd_conf['input_w'], ssd_conf['input_c']])
    labels, bboxes, confidence, endpoints = ssd_net.predict(input_img)

    # Compute loss and put them to tf.collections.LOSS and other loss.
    input_labels = tf.placeholder(tf.int8, [None, 1])
    input_bboxes = tf.placeholder(tf.int8, [None, 4])
    total_loss = ssd_net.loss(labels=labels, bboxes=bboxes, glabels=input_labels, gbboxes=input_bboxes)

    # load traning config.
    trainConf = script_conf.loadTrainConf()

    # Create optimizer.
    optimizer = utils.getOptimizer(trainConf)

    # Train model using optimizer.
    train_op = optimizer.minimize(total_loss)

    # Prepare training data.
    dataset = dt.DataSet(trainConf['dataset_path'], trainConf['batch_size'])
    imgs, sizes, glabels, gbboxes = dataset.getNext()

    with tf.Session() as ss:
        for i in range(trainConf['epoch_num']):
            ss.run(train_op, feed_dict={input_img: imgs, input_labels: glabels, input_bboxes: gbboxes,
                                        labels: labels, bboxes: bboxes})
    #
    # # Save trained parameter to file.
    # tl.files.save_npz(ssd_net.all_params, name='model.npz')

if __name__ == '__main__':
    tf.app.run()