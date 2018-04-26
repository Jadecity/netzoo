"""
Scripts for training a ssd network.
"""

import tensorflow as tf
import tensorlayer as tl
import tensorboard as tb
import network.ssdConf as ssdConf
import common.dataset as dt
import network.ssdNet as ssdNet
import network.mobileNet as mobileNet

def main():
    """
    Main script for training.
    :return: None.
    """
    # load traning config
    tf.flags.DEFINE_integer('batch_size', 1)
    trainConf = tf.flags.FLAGS
    pass

    # prepare training data
    dataset_path = '/home/autel/data/INRIAPerson/Train/tfrecords/pos'
    dataset = dt.DataSet(dataset_path, trainConf.batch_size)
    img_batch, size_batch, labels_batch, bboxes_batch = dataset.getNext()

    # load SSD config
    ssd_conf = ssdConf()

    # create network
    ft_extractor = mobileNet.MobileNet()
    ssd_net = ssdNet.SSDNet(ssd_conf, ft_extractor)
    

    pass

if __name__ == '__main__':
    tf.app.run()