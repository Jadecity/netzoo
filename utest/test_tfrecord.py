import os
import tensorflow as tf
import json
import tensorlayer.visualize as vis
import numpy as np
import common.utils as utils


def _int64List_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.flatten()))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature_list(value):
    """
    Return a list of feature.
    :param value: np array, shape[n, k]
    """
    return [_int64List_feature(v) for v in value]

# _bytes is used for string/char values
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def test_tfrecord_create(repeat_num):
    writer = tf.python_io.TFRecordWriter('/tmp/tmp.tfrecords')

    for i in range(repeat_num):
        if i == 0:
            img_size = np.ones([40, 50], dtype=np.int64)
            bboxes = np.array([[1,2,3], [4,5,6]], dtype=np.int64)
        else:
            img_size = np.ones([20, 30],dtype=np.int64)
            bboxes = np.array([[1, 2, 3]],dtype=np.int64)
            
        feature = {
            'name': _bytes_feature(tf.compat.as_bytes('abc')),
            'image': _bytes_feature(np.array([6,6,6], dtype=np.int64).tobytes()),
            'bsize': _int64List_feature(bboxes),
            'label': _int64List_feature(np.array([5]))
            # 'bboxes': _int64List_feature(bboxes)
        }

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Writing the serialized example.
        example_str = example.SerializeToString()
        writer.write(example_str)
        
    writer.close()

def test_tfrecord_decode():
    file_queue = tf.train.string_input_producer(['/tmp/tmp.tfrecords'])
    reader = tf.TFRecordReader()
    _, example = reader.read(file_queue)

    feature = {
        # 'image': tf.FixedLenFeature([], dtype=tf.string),
        'bsize': tf.VarLenFeature(tf.int64)
    }

    context_parsed = tf.parse_single_example(serialized=example,
                                             features=feature)

    # image = tf.decode_raw(context_parsed['image'], tf.uint8)
    size = context_parsed['bsize']
    size = tf.sparse_tensor_to_dense(size)

    return size

def test_tfrecord_decode_v2():
    """
            Parse tfrecord example.
            :param exam: one example instance
            :return: image, size, labels, bboxes
            """
    file_queue = tf.train.string_input_producer(['/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/tfrecords/1.tfrecords'])
    reader = tf.TFRecordReader()
    _, example = reader.read(file_queue)

    context_feature = {
        'image': tf.FixedLenFeature([], tf.string),
        'size': tf.FixedLenFeature([3], dtype=tf.int64),
        'bbox_num': tf.FixedLenFeature([1], dtype=tf.int64)
    }

    varlen_feature = {
        'bboxes': tf.FixedLenSequenceFeature([5], dtype=tf.int64)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=example,
        context_features=context_feature,
        sequence_features=varlen_feature
    )

    image = context_parsed['image']
    size = context_parsed['size']
    bbox_num = context_parsed['bbox_num']
    bbox_num = tf.squeeze(bbox_num)
    bboxes = sequence_parsed['bboxes']

    labels = tf.slice(bboxes, [0, 0], [tf.cast(bbox_num, dtype=tf.int32), 1])
    # labels = tf.slice(bboxes, [0, 0], [1, 1])
    # bboxes = tf.slice(bboxes, [0, 1], [tf.cast(bbox_num, dtype=tf.int32), 4])
    # labels = tf.squeeze(labels)
    #
    # labels = utils.makeOneHot(labels, 20)

    # labels = bbox_num
    return image, size, labels, bboxes

def parse_func(example):
    """
    Parse tfrecord example.
    :param exam: one example instance
    :return: image, size, labels, bboxes
    """
    feature = {
        'name': tf.FixedLenFeature([], dtype=tf.string),
        'image': tf.FixedLenFeature([], dtype=tf.string),
        'bsize': tf.VarLenFeature(tf.int64),
        'label': tf.FixedLenFeature([1], dtype=tf.int64)
    }

    context_parsed = tf.parse_single_example(serialized=example,
                                             features=feature)

    # image = tf.decode_raw(context_parsed['image'], tf.uint8)
    size = context_parsed['bsize']
    size = tf.sparse_tensor_to_dense(size)

    image = tf.decode_raw(context_parsed['image'], tf.uint8)
    label = context_parsed['label']
    name = context_parsed['name']

    return name, image, size, label

if __name__ == '__main__':
    test_tfrecord_create(10)

    # size = test_tfrecord_decode()
    dataset = tf.data.TFRecordDataset(['/tmp/tmp.tfrecords'])
    dataset = dataset.map(parse_func)
    dataset = dataset.padded_batch(2, padded_shapes=([], [None], [None], tf.TensorShape([None])))
    itr = dataset.make_one_shot_iterator()
    names, imgs, size_batch, label = itr.get_next()

    with tf.Session() as sess:
        tf.train.start_queue_runners(sess=sess)

        names = sess.run([names[0]])
        for name in names:
            print(name)

    # test TFRecordDataset
    # rcd_files = ['/tmp/tmp.tfrecords']

    # dataset = tf.data.TFRecordDataset(rcd_files)
    # dataset = dataset.map(map_func=parse_func)
    # # dataset = dataset.batch(2)
    # dataset = dataset.padded_batch(2, padded_shapes=([None], tf.TensorShape([None,3])))
    # itr = dataset.make_one_shot_iterator()
    # labels, frames = itr.get_next()
    # d1 = tf.slice(frames[0], [0, 0], [tf.cast(frames[0][0][1], dtype=tf.int32), 1])
    # with tf.Session() as ss:
    #     print(ss.run(d1))
    #
    # image, size, labels, bboxes = test_tfrecord_decode_v2()
    # a = tf.cast(labels, tf.int32)
    # with tf.Session() as sess:
    #     tf.train.start_queue_runners(sess=sess)
    #     a, image, size, labels, bboxes = sess.run((a, image, size, labels, bboxes))
    #     print(labels)
    #     print(np.shape(bboxes))
    #     print(a)