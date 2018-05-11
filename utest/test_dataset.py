import common.dataset as dt
import tensorflow as tf
import common.config as script_conf
import numpy as np
import matplotlib.pyplot as plt
import common.utils as utils
import json

if __name__ == '__main__':
    # load traning config.
    trainConf = script_conf.loadTrainConf()

    dataset = dt.DataSet(path = trainConf['dataset_path'],
                         batchsize = 10,
                         class_num = trainConf['class_num'])

    # img_batch, size_batch, \
    img_name_batch, img_batch, sizes_batch, class_id_batch, box_num_batch, labels_batch, bboxes_batch = dataset.getNext()
    class_id_batch_hot = tf.one_hot(class_id_batch, trainConf['class_num'] + 1)
    class_id_batch_hot = class_id_batch_hot[:, 1:]
    # labels_mask = tf.greater(labels_batch, 0)
    # labels_batch = tf.one_hot(labels_batch, 20)
    # labels_batch = tf.boolean_mask(labels_batch, labels_mask, axis=0)
    #
    # bboxes_batch = tf.boolean_mask(bboxes_batch, labels_mask, axis=0)

    # img_fig = plt.imshow(np.zeros([100, 100]))

    with  open('/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/classes.json') as label_name_file:
        class_dict = json.load(label_name_file)

    with tf.Session() as ss:
        ss.run(dataset._itr.initializer)
        for _ in range(1):
            img_name_batch, img_batch, sizes_batch, class_id_batch, box_num_batch, labels_batch, bboxes_batch = ss.run((img_name_batch, img_batch, sizes_batch, class_id_batch, box_num_batch, labels_batch, bboxes_batch))
            # print(one_hot_labels)
            # print(img_batch, labels_batch, bboxes_batch)

            for i in range(10):
                shape = [sizes_batch[i, 1], sizes_batch[i, 0], sizes_batch[i, 2]]
                img = img_batch[i]
                # img = np.fromstring(img, dtype=np.uint8)
                img.shape = shape

                box_num = box_num_batch[i]
                if i > 0:
                    start_idx = np.sum(box_num_batch[:i])
                else:
                    start_idx = 0
                bboxes = bboxes_batch[start_idx:start_idx + box_num]

                img_name = img_name_batch[i]

                print(img_name)
                print(class_id_batch)

                # utils.visulizeBBox(img, bboxes)

                # resizer = utils.createResizePreprocessor({'dest_size':np.array([300, 300])})
                # img, size, bboxes = resizer(img, sizes_batch[i], bboxes)

                # utils.visulizeBBox(img, bboxes, True)
                utils.visulizeClass(img, class_id_batch[i], class_dict)
                plt.waitforbuttonpress()
                # plt.close('all')
                #
                # img_d = trans.rescale(img, 0.5)
                # plt.imshow(img_d)
                # plt.draw()
                # plt.waitforbuttonpress()


