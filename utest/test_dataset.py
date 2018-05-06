import common.dataset as dt
import tensorflow as tf
import tensorlayer as tl
import common.config as script_conf
import numpy as np

if __name__ == '__main__':
    dataset_path = '/home/autel/data/INRIAPerson/Train/tfrecords/pos'
    # load traning config.
    trainConf = script_conf.loadTrainConf()

    dataset = dt.DataSet(path = trainConf['dataset_path'],
                         batchsize = trainConf['batch_size'],
                         class_num = trainConf['class_num'])

    img_batch, size_batch, labels_batch, bboxes_batch = dataset.getNext()
    print(labels_batch)
    with tf.Session() as ss:
        img_batch, size_batch, labels_batch, bboxes_batch = ss.run((img_batch, size_batch, labels_batch, bboxes_batch))
        print(labels_batch)
        # tl.visualize.frame(img_batch)