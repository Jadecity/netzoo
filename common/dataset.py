"""
Utility tools to manage tfrecord dataset.
"""
import os.path
import tensorflow as tf
import glob

class DataSet:

    def __init__(self, path, parser=None, batchsize=1):
        self._dataset = None
        self.createDataSet(path, batchsize, parser)

    def _parse_func(self, example):
        """
        Parse tfrecord example.
        :param exam: one example instance
        :return: image, size, labels, bboxes
        """
        feature = {
            'image': tf.FixedLenFeature([], tf.string),
            'size': tf.VarLenFeature(tf.int64),
            'labels': tf.VarLenFeature(tf.int64),
            'bboxes': tf.VarLenFeature(tf.int64)
        }

        parsed_features = tf.parse_single_example(example, feature)

        image = parsed_features['image']
        size = tf.sparse_tensor_to_dense(parsed_features['size'])
        labels = tf.sparse_tensor_to_dense(parsed_features['labels'])
        bboxes = tf.sparse_tensor_to_dense(parsed_features['bboxes'])

        return image, size, labels, bboxes

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
        dataset = dataset.batch(batchsize)
        self._dataset = dataset
        self._itr = dataset.make_one_shot_iterator()
        return

    def getNext(self):
        """
        Get next batch or element according to batch settings when create dataset.
        :return:
        """
        img_batch, size_batch, labels_batch, bboxes_batch = self._itr.get_next()
        return img_batch, size_batch, labels_batch, bboxes_batch

