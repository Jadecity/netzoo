import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from collections import namedtuple

def getOptimizer(opt_conf):
    """
    Create an optimizer using config.
    :param opt_conf: optimizer configuration
    :return: tf.train.Optimizer object or None
    """
    optimizer = None

    if opt_conf['optimizer'] == 'AdamOptimizer':
        optimizer = tf.train.AdamOptimizer(learning_rate=opt_conf['learning_rate'],
                                           epsilon=opt_conf['epsilon'])

    return optimizer

def makeOneHot(labels, class_num):
    """
    Tranform labels to one hot labels according to class number.
    :param labels: labels of one example
    :param class_num: number of total classes.
    :return: transformed label batch.
    """
    transformed = tf.one_hot(labels, depth=class_num, dtype=tf.int32)
    return transformed

def jaccardIndex(gbbox, bboxes):
    """
    Compute jaccard index between bbox1 and bbox2.
    :param gbbox: ground truth bouding box, [x, y, width, height]
    :param bboxes: bouding box 1, shape[width, height, anchor_num, 4], 4 is [x, y, width, height]
    :return: jaccard index, shape[width, height, anchor_num]
    """

    BBox = namedtuple('BBox', ['x', 'y', 'width', 'height'])
    gbbox = BBox(x=gbbox[0], y=gbbox[1], width=gbbox[2], height=gbbox[3])
    bboxes = BBox(x=bboxes[:, :, :, 0], y=bboxes[:, :, :, 1], width=bboxes[:, :, :, 2], height=bboxes[:, :, :, 3])

    xmin = tf.minimum(tf.subtract(gbbox.x, tf.divide(gbbox.width, 2.0)),
                      tf.subtract(bboxes.x, tf.divide(bboxes.width, 2.0)))
    xmax = tf.maximum(tf.add(gbbox.x, tf.divide(gbbox.width, 2.0)),
                      tf.add(bboxes.x, tf.divide(bboxes.width, 2.0)))
    ymin = tf.minimum(tf.subtract(gbbox.y, tf.divide(gbbox.height, 2.0)),
                      tf.subtract(bboxes.y, tf.divide(bboxes.height, 2.0)))
    ymax = tf.maximum(tf.add(gbbox.y, tf.divide(gbbox.height, 2.0)),
                      tf.add(bboxes.y, tf.divide(bboxes.height, 2.0)))

    width = tf.subtract(tf.subtract(xmax, xmin), tf.add(gbbox.width, bboxes.width))
    width = tf.minimum(tf.constant(0, dtype=tf.float32), width)
    height = tf.subtract(tf.subtract(ymax, ymin), tf.add(gbbox.height, bboxes.height))
    height = tf.minimum(tf.constant(0, dtype=tf.float32), height)

    intersect_area = tf.multiply(width, height)
    union_area = tf.subtract(tf.add(tf.multiply(gbbox.width, gbbox.height),
                                    tf.multiply(bboxes.width, bboxes.height)),
                             intersect_area)

    # ss = tf.Session()
    # print(ss.run([gbbox]))
    # ss.close()

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

    return tf.greater(overlap, 0.5)




def visualizeAnchors(anchors, gconf, gbboxes):
    ftmap_conf = gconf['featuremaps']
    layer_num = len(anchors)
    for layer_i in range(0, layer_num):
        shape = np.shape(anchors[ftmap_conf[layer_i].layer_name])
        print(shape)

        cur_anchors = anchors[ftmap_conf[layer_i].layer_name]
        print(np.shape(cur_anchors))
        # show each anchor in the same image
        img = np.zeros([shape[0], shape[1]])
        fig, ax = plt.subplots(1)

        for i in range(shape[0]):
            for j in range(shape[1]):
                ax.imshow(img)
                for k in range(shape[2]):
                    cx, cy, w, h = cur_anchors[i, j, k, :]
                    cx *= np.int32(shape[0])
                    cy *= np.int32(shape[1])
                    w *= np.int32(shape[0])
                    h *= np.int32(shape[1])
                    cx = max(0, cx - w/2)
                    cy = max(0, cy - h/2)

                    # Create a Rectangle patch
                    rect = patches.Rectangle((cx, cy), w, h, linewidth=1, edgecolor='r', facecolor='none')

                    # Add the patch to the Axes
                    ax.add_patch(rect)

                    print('i:%d, j:%d, cx: %d, cy: %d, w:%d, h:%d' % (i, j, cx, cy, w, h))

                cx, cy, w, h = gbboxes[:]
                cx *= np.int32(shape[0])
                cy *= np.int32(shape[1])
                w *= np.int32(shape[0])
                h *= np.int32(shape[1])
                cx = max(0, cx - w / 2)
                cy = max(0, cy - h / 2)
                # Create a Rectangle patch
                rect = patches.Rectangle((cx, cy), w, h, linewidth=1, edgecolor='b', facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)

                # plt.waitforbuttonpress(timeout=1)
                plt.pause(0.01)
                ax.clear()

        plt.close()

def visualizeOverlap(anchors, gconf, gbboxes):
    ftmap_conf = gconf['featuremaps']
    layer_num = len(anchors)

    gbboxes = tf.constant(gbboxes, dtype=tf.float32)

    ss = tf.Session()
    for layer_i in range(1, layer_num):
        shape = np.shape(anchors[ftmap_conf[layer_i].layer_name])
        print(shape)

        cur_anchors = anchors[ftmap_conf[layer_i].layer_name]
        cur_anchors = tf.constant(cur_anchors, dtype=tf.float32)
        overlap = jaccardIndex(gbboxes, cur_anchors)

        overlap = ss.run(overlap)

        # print overlap
        print(overlap)

        plt.imshow(np.zeros([10,10]))
        plt.waitforbuttonpress()

