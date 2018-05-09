import common.dataset as dt
import tensorflow as tf
import tensorlayer as tl
import common.config as script_conf
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # load traning config.
    trainConf = script_conf.loadTrainConf()

    dataset = dt.DataSet(path = trainConf['dataset_path'],
                         batchsize = 4,
                         class_num = trainConf['class_num'])

    # img_batch, size_batch, \
    img_batch, sizes_batch, box_num_batch, labels_batch, bboxes_batch = dataset.getNext()
    labels_mask = tf.greater(labels_batch, 0)
    one_hot_labels = tf.one_hot(labels_batch, 20)
    labels_batch = tf.boolean_mask(one_hot_labels, labels_mask, axis=0)


    with tf.Session() as ss:
        for _ in range(1):
            # imgs, sizes, \img_batch, size_batch,
            # imgs, sizes, box_num, labels, bboxes = ss.run((img_batch, sizes_batch, box_num_batch, labels_batch, bboxes_batch))
            one_hot_labels,labels_batch,labels_mask = ss.run((one_hot_labels, labels_batch, labels_mask))
            print(one_hot_labels)
            print(labels_batch)
            print(labels_mask)

        # # for img in img_batch:
        # img_batch = np.fromstring(img_batch, dtype=np.uint8)
        # img_batch.shape = [size_batch[1], size_batch[0], size_batch[2]]
        #
        # tl.visualize.frame(img_batch, saveable=False)
        # plt.waitforbuttonpress()
