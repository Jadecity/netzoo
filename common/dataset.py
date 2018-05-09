"""
Utility tools to manage tfrecord dataset.
"""
import os.path
import tensorflow as tf
import glob
import common.utils as utils

class DataSet:

    def __init__(self, path, class_num, parser=None, batchsize=1):
        self._dataset = None
        self._class_num = class_num
        self.createDataSet(path, batchsize, parser)

    def _parse_func(self, example):
        """
        Parse tfrecord example.
        :param exam: one example instance
        :return: image, size, labels, bboxes
        """

        feature = {
            'image': tf.FixedLenFeature([], dtype=tf.string),
            'size': tf.FixedLenFeature([3], tf.int64),
            'labels': tf.VarLenFeature(tf.int64),
            'bbox_num': tf.FixedLenFeature([1], tf.int64),
            'bboxes': tf.VarLenFeature(tf.int64)
        }

        context_parsed = tf.parse_single_example(serialized=example,
                                                 features=feature)

        image = tf.decode_raw(context_parsed['image'], tf.uint8)
        size = context_parsed['size']
        labels = tf.sparse_tensor_to_dense(context_parsed['labels'])

        bbox_num = context_parsed['bbox_num']
        boxes_shape = tf.stack([bbox_num[0], 4])
        bboxes = tf.sparse_tensor_to_dense(context_parsed['bboxes'])
        bboxes = tf.reshape(bboxes, shape=boxes_shape)

        return image, size, bbox_num, labels, bboxes

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

        # rcd_files = glob.glob(os.path.join(path, '*.tfrecords'))
        rcd_files = ['/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/tfrecords/1.tfrecords']
        if len(rcd_files) == 0:
            raise FileNotFoundError('No TFRecords file found in %s!' % path)

        dataset = tf.data.TFRecordDataset(rcd_files)
        dataset = dataset.map(map_func=parser)
        padding_shape = ([None], [None], [None], tf.TensorShape([None]), tf.TensorShape([None, 4]))#, tf.TensorShape([None]))
        dataset = dataset.padded_batch(batchsize, padded_shapes=padding_shape)
        self._dataset = dataset
        self._itr = dataset.make_one_shot_iterator()
        return

    def getNext(self):
        """
        Get next batch or element according to batch settings when create dataset.
        :return:
        """

        # image_batch, size_batch, bbox_num, \
        imgs, sizes, box_nums, labels, bboxes = self._itr.get_next() #, bbox_num_batch, labels_batch, bboxes_batch

        # print(size_batch.get_shape())
        # for bbox_num, bboxes in zip(bbox_num_batch, bboxes_batch):
        #     labels = tf.slice(bboxes, [0, 0], [tf.cast(bbox_num, dtype=tf.int32), 1])
        #     bboxes = tf.slice(bboxes, [0, 1], [tf.cast(bbox_num, dtype=tf.int32), 4])
        #     labels = tf.squeeze(labels)
        #     labels = utils.makeOneHot(labels, self._class_num)
        #
        #     labels_batch.append(labels)
        #     boxes_batch.append(bboxes)

        # image_batch, size_batch, \


        return imgs, sizes, box_nums, labels, bboxes

