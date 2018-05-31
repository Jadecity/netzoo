"""
Utility tools to manage tfrecord dataset.
"""
import os.path
import glob
import tensorflow as tf
import common.config as script_conf
import matplotlib.pyplot as plt
import common.utils as utils
import numpy as np


class CifarDataSet:

    def __init__(self, path, class_num, mean_img, parser=None, batchsize=1):
        self._dataset = None
        self._class_num = class_num
        self._mean_img = np.load(mean_img)
        self.createDataSet(path, batchsize, parser)

    def _parse_func(self, example):
        """
        Parse tfrecord example.
        :param exam: one example instance
        :return: image, size, labels, bboxes
        """

        feature = {
            'image_name': tf.FixedLenFeature([], dtype=tf.string),
            'image': tf.FixedLenFeature([], dtype=tf.string),
            'size': tf.FixedLenFeature([3], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'label_name': tf.FixedLenFeature([], dtype=tf.string)
        }

        context_parsed = tf.parse_single_example(serialized=example,
                                                 features=feature)

        image = tf.decode_raw(context_parsed['image'], tf.uint8)
        size = context_parsed['size']
        image = tf.reshape(image, size)
        image = tf.cast(image, dtype=tf.float32)
        image -= self._mean_img

        image_name = context_parsed['image_name']
        label_id = context_parsed['label']
        label_name = context_parsed['label_name']

        return image_name, image, size, label_id, label_name

    def createDataSet(self, path, batchsize=1, parser=None):
        """
        Create a tfrecrod dataset, all tfrecords should be in path directory.
        :param path: directory where tfrecords files live.
        :param parser: parse function for each TFRecord.
        :return: TFRecordDataset object.
        """

        if not os.path.exists(path):
            raise FileNotFoundError('Path %s not exist!' % path)

        if None == parser:
            parser = self._parse_func

        rcd_files = glob.glob(os.path.join(path, '*.tfrecords'))
        if len(rcd_files) == 0:
            raise FileNotFoundError('No TFRecords file found in %s!' % path)

        dataset = tf.data.TFRecordDataset(rcd_files)
        dataset = dataset.map(map_func=parser)
        # padding_shape = ([], [None, None, None], [None], [], [])
        # dataset = dataset.padded_batch(batchsize, padded_shapes=padding_shape)
        dataset = dataset.batch(batchsize)
        dataset = dataset.shuffle(buffer_size=200)
        dataset.prefetch(buffer_size=1000)
        self._dataset = dataset
        self._itr = dataset.make_initializable_iterator()

        return

    def getNext(self):
        """
        Get next batch or element according to batch settings when create dataset.
        :return:
        """

        img_name_batch, img_batch, size_batch, class_id_batch, label_name_batch = self._itr.get_next()

        return img_name_batch, img_batch, size_batch, class_id_batch, label_name_batch

if __name__ == '__main__':
    # test CifarDataset
    # load traning config.
    trainConf = script_conf.loadTrainConf()

    dataset = CifarDataSet(path = trainConf['dataset_path'],
                         batchsize = trainConf['batch_size'],
                         class_num = trainConf['class_num'],
                           mean_img=trainConf['mean_img'])

    # img_batch, size_batch, \
    img_name_batch, img_batch, sizes_batch, class_id_batch, label_name_batch = dataset.getNext()
    class_id_batch_hot = tf.one_hot(class_id_batch, trainConf['class_num'])

    with tf.Session() as ss:
        ss.run(dataset._itr.initializer)
        for _ in range(1):
            img_name_batch, img_batch, sizes_batch, class_id_batch,label_name_batch = ss.run((img_name_batch, img_batch, sizes_batch, class_id_batch,label_name_batch))

            for i in range(trainConf['batch_size']):
                shape = [sizes_batch[i, 1], sizes_batch[i, 0], sizes_batch[i, 2]]
                img = img_batch[i]
                img = np.fromstring(img, dtype=np.uint8)
                img.shape = shape

                img_name = img_name_batch[i]

                print(img_name)
                print(class_id_batch[i])

                utils.visulizeClassByName(img, label_name_batch[i], True)

                # resizer = utils.createResizePreprocessor({'dest_size':np.array([300, 300])})
                # img, size, bboxes = resizer(img, sizes_batch[i], bboxes)

                # utils.visulizeBBox(img, bboxes, True)
                # utils.visulizeClass(img, class_id_batch[i], class_dict, hold=True)
                plt.waitforbuttonpress()