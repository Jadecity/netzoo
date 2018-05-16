"""
Utility tools to manage tfrecord dataset.
"""
import os.path
import tensorflow as tf
import glob


class PascalDataSet:

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
            'image_name': tf.FixedLenFeature([], dtype=tf.string),
            'image': tf.FixedLenFeature([], dtype=tf.string),
            'size': tf.FixedLenFeature([3], tf.int64),
            'class': tf.FixedLenFeature([], tf.int64),
            'labels': tf.VarLenFeature(tf.int64),
            'bbox_num': tf.FixedLenFeature([], tf.int64),
            'bboxes': tf.VarLenFeature(tf.int64)
        }

        context_parsed = tf.parse_single_example(serialized=example,
                                                 features=feature)

        image = tf.decode_raw(context_parsed['image'], tf.uint8)
        size = context_parsed['size']
        image = tf.reshape(image, size)
        # image = tf.cast(image, tf.float32)
        # img_mean = tf.fill([224, 224, 3], 125.0)
        # image = tf.subtract(image, img_mean)
        labels = tf.sparse_tensor_to_dense(context_parsed['labels'])

        bbox_num = context_parsed['bbox_num']
        boxes_shape = tf.stack([bbox_num, 4])
        bboxes = tf.sparse_tensor_to_dense(context_parsed['bboxes'])
        bboxes = tf.reshape(bboxes, shape=boxes_shape)
        image_name = context_parsed['image_name']

        class_id = context_parsed['class']

        return image_name, image, size, class_id, bbox_num, labels, bboxes

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
        # rcd_files = ['/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/tfrecords/1.tfrecords']
        if len(rcd_files) == 0:
            raise FileNotFoundError('No TFRecords file found in %s!' % path)

        dataset = tf.data.TFRecordDataset(rcd_files)
        dataset = dataset.map(map_func=parser)
        padding_shape = ([], [None, None, None], [None], [], [], tf.TensorShape([None]), tf.TensorShape([None, 4]))#, tf.TensorShape([None]))
        dataset = dataset.padded_batch(batchsize, padded_shapes=padding_shape)
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

        img_name_batch, img_batch, size_batch, class_id_batch, box_num_batch, labels_batch, bboxes_batch = self._itr.get_next() #, bbox_num_batch, labels_batch, bboxes_batch

        # remove label paddings and create one-hot labels
        labels_mask = tf.greater(labels_batch, 0)
        one_hot_labels = tf.one_hot(labels_batch, 20)
        labels_batch = tf.boolean_mask(one_hot_labels, labels_mask, axis=0)

        # remove bbox paddings
        bboxes_batch = tf.boolean_mask(bboxes_batch, labels_mask, axis=0)


        return img_name_batch, img_batch, size_batch, class_id_batch, box_num_batch, labels_batch, bboxes_batch

