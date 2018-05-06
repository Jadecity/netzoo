import tensorflow as tf
from collections import namedtuple

def getOptimizer(opt_conf):
    """
    Create an optimizer using config.
    :param opt_conf: optimizer configuration
    :return: tf.train.Optimizer object or None
    """
    optimizer = None

    if opt_conf['opt_name'] == 'AdamOptimizer':
        optimizer = tf.train.AdamOptimizer(learning_rate=opt_conf['learning_rate'],
                                           epsilon=opt_conf['epsilon'])

    return optimizer

def makeOneHot(labels_batch, class_num):
    """
    Tranform labels to one hot labels according to class number.
    :param labels_batch: batch of labels.
    :param class_num: number of total classes.
    :return: transformed label batch.
    """
    batch_transformed = tf.one_hot(labels_batch, depth=class_num, dtype=tf.int64)
    return batch_transformed

def jaccardIndex(gbbox, bbox2):
    """
    Compute jaccard index between bbox1 and bbox2.
    :param gbbox: ground truth bouding box, [x, y, width, height]
    :param bbox2: bouding box 1, shape[width, height, anchor_num, 4], 4 is [x, y, width, height]
    :return: jaccard index, shape[width, height, anchor_num]
    """
    BBox = namedtuple('BBox', ['x', 'y', 'width', 'height'])
    gbbox = BBox(x=gbbox[0], y=gbbox[1], width=gbbox[2], height=gbbox[3])
    bbox2 = BBox(x=bbox2[:,:,:, 0], y=bbox2[:,:,:, 1], width=bbox2[:,:,:,2], height=bbox2[:,:,:,3])

    xmin = tf.minimum(tf.subtract(gbbox.x, tf.divide(gbbox.width, 2)),
                      tf.subtract(bbox2.x, tf.divide(bbox2.width, 2)))
    xmax = tf.maximum(tf.add(gbbox.x, tf.divide(gbbox.width, 2)),
                      tf.add(bbox2.x, tf.divide(bbox2.width, 2)))
    ymin = tf.minimum(tf.subtract(gbbox.y, tf.divide(gbbox.height, 2)),
                      tf.subtract(bbox2.y, tf.divide(bbox2.height, 2)))
    ymax = tf.maximum(tf.add(gbbox.y, tf.divide(gbbox.height, 2)),
                      tf.add(bbox2.y, tf.divide(bbox2.height, 2)))

    width = tf.subtract(tf.subtract(xmax, xmin), tf.add(gbbox.width, bbox2.width))
    width = tf.minimum(tf.constant(0, dtype=tf.float32), width)
    height = tf.subtract(tf.subtract(ymax, ymin), tf.add(gbbox.height, bbox2.height))
    height = tf.minimum(tf.constant(0, dtype=tf.float32), height)

    intersect_area = tf.multiply(width, height)
    union_area = tf.subtract(tf.add(tf.multiply(gbbox.width, gbbox.height),
                                    tf.multiply(bbox2.width, bbox2.height)),
                             intersect_area)

    return tf.divide(intersect_area, union_area)

def smoothL1(x):
    """
    Compute l1 smooth for each element in tensor x.
    :param x: input tensor.
    :return: l1 smooth of x.
    """
    fx = tf.where(tf.less(tf.abs(x), 1.0),
                 tf.multiply(tf.square(x), 0.5),
                 tf.subtract(tf.abs(x), 0.5))
    return fx

def positiveMask(overlap):
    """
    Compute boolean mask for positive examples.
    :param overlap: shape[width, height, anchor_num, 4]
    :return: boolean mask of shape [width, height, anchor_num, 4]
    """

    return tf.where(tf.greater(overlap, 0.5))