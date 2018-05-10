import common.dataset as dt
import tensorflow as tf
import tensorlayer as tl
import common.config as script_conf
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as trans
import common.utils as utils

if __name__ == '__main__':
    # load traning config.
    trainConf = script_conf.loadTrainConf()

    dataset = dt.DataSet(path = trainConf['dataset_path'],
                         batchsize = 10,
                         class_num = trainConf['class_num'])

    # img_batch, size_batch, \
    img_name_batch, img_batch, sizes_batch, box_num_batch, labels_batch, bboxes_batch = dataset.getNext()
    # labels_mask = tf.greater(labels_batch, 0)
    # labels_batch = tf.one_hot(labels_batch, 20)
    # labels_batch = tf.boolean_mask(labels_batch, labels_mask, axis=0)
    #
    # bboxes_batch = tf.boolean_mask(bboxes_batch, labels_mask, axis=0)

    # img_fig = plt.imshow(np.zeros([100, 100]))
    with tf.Session() as ss:
        for _ in range(1):
            img_name_batch, img_batch, sizes_batch,box_num_batch, labels_batch, bboxes_batch = ss.run((img_name_batch, img_batch, sizes_batch, box_num_batch, labels_batch, bboxes_batch))
            # print(one_hot_labels)
            # print(img_batch, labels_batch, bboxes_batch)

            for i in range(10):
                shape = [sizes_batch[i, 1], sizes_batch[i, 0], sizes_batch[i, 2]]
                img = img_batch[i, :np.product(shape)]
                img = np.fromstring(img, dtype=np.uint8)
                img.shape = shape

                box_num = box_num_batch[i][0]
                if i > 0:
                    start_idx = np.sum(box_num_batch[:i])
                else:
                    start_idx = 0
                bboxes = bboxes_batch[start_idx:start_idx + box_num]

                img_name = img_name_batch[i]
                print(img_name)

                utils.visulizeBBox(img, bboxes)
                plt.waitforbuttonpress()

                resizer = utils.createResizePreprocessor({'dest_size':np.array([300, 300])})
                img, size, bboxes = resizer(img, sizes_batch[i], bboxes)

                utils.visulizeBBox(img, bboxes)
                plt.waitforbuttonpress()
                plt.close('all')
                #
                # img_d = trans.rescale(img, 0.5)
                # plt.imshow(img_d)
                # plt.draw()
                # plt.waitforbuttonpress()


