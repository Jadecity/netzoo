import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from collections import namedtuple
import skimage.transform as trans
import cv2
import glob
from os import path

def int64List_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.flatten()))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# _bytes is used for string/char values
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

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
                ax.clear()

        plt.close('all')

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

def visulizeBBox(img, bboxes, hold=False):
    if not hold:
        fig, ax = plt.subplots(1)
    else:
        ax = plt.gca()

    ax.clear()
    ax.imshow(img)
    for box in bboxes:
        # Create a Rectangle patch
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.draw()

def visulizeClass(img, label, label_dict, hold=False):
    """
    Visualize image and its class name.
    :param img: image.
    :param label: class label in number.
    :param label_dict: lable name and its number dictionary.
    :return: None.
    """
    # label = np.argmax(label) + 1
    class_name = ''
    for name in label_dict.keys():
        if label == label_dict[name]:
            class_name = name
            break

    # Show image and its label.
    if not hold:
        fig, ax = plt.subplots(1)
    else:
        ax = plt.gca()

    ax.clear()
    ax.imshow(img)

    # Add the patch to the Axes
    ax.text(10, 10, class_name, color='r', bbox=dict(facecolor='green', alpha=0.5))

    plt.draw()

def visulizeClassV2(img, label, label_name_list, hold=False):
    """
    Visualize image and its class name.
    :param img: image.
    :param label: class label in number.
    :param label_name_list: a list of lable name.
    :return: None.
    """
    # label = np.argmax(label) + 1
    class_name = label_name_list[label]

    # Show image and its label.
    if not hold:
        fig, ax = plt.subplots(1)
    else:
        ax = plt.gca()

    ax.clear()
    ax.imshow(img)

    # Add the patch to the Axes
    ax.text(10, 10, class_name, color='r', bbox=dict(facecolor='green', alpha=0.5))

    plt.draw()

def visulizeClassByName(img, label_name, hold=False):
    """
    Visualize image and its class name.
    :param img: image.
    :param label_name: single label name
    :return: None.
    """
    # label = np.argmax(label) + 1
    class_name = label_name

    # Show image and its label.
    if not hold:
        fig, ax = plt.subplots(1)
    else:
        ax = plt.gca()

    ax.clear()
    ax.imshow(img)

    # Add the patch to the Axes
    ax.text(10, 10, class_name, color='r', bbox=dict(facecolor='green', alpha=0.5))

    plt.draw()

class ResizePreprocessor:
    def __init__(self, conf):
        """
        This version, destination size should be square.
        :param conf:
        """

        self._dest_size = conf['dest_size']

    def __call__(self, img, size, bboxes):
        """
        Resize to dest size but keep all bounding boxes.
        Resize with ratio kept is first applied, then crop to dest size.
        Bbox positions are adjusted according to the new size.
        :param img: ndarray [MxNxC]
        :param size: ndarray [w, h, c]
        :param bboxes: ndarray, [Kx(x1, y1, x2, y2)]
        :return: new img, size, bboxes
        """
        w, h = size[0], size[1]
        wd, hd = self._dest_size[0], self._dest_size[1]
        ratio = np.float(w)/np.float(h)
        if ratio > 1:
            hm = hd
            wm = np.int(hm * ratio)
        else:
            wm = wd
            hm = np.int(wm / ratio)

        # Scale with ratio kept.
        # img_d = trans.resize(image=img, output_shape=([hm, wm]), preserve_range=True)
        img_d = cv2.resize(img, (wm, hm))

        # Scale bboxes
        w_s, h_s = wm / w, hm / h
        bboxes[:, 0] = (bboxes[:, 0] * w_s).astype(np.int)
        bboxes[:, 2] = (bboxes[:, 2] * w_s).astype(np.int)
        bboxes[:, 1] = (bboxes[:, 1] * h_s).astype(np.int)
        bboxes[:, 3] = (bboxes[:, 3] * h_s).astype(np.int)

        # Crop center area
        cx, cy = wm / 2, hm / 2
        x, y = np.int(cx - wd/2), np.int(cy - hd/2)
        min_x = np.min(bboxes[:, 0])
        min_y = np.min(bboxes[:, 1])
        max_x = np.max(bboxes[:, 2])
        max_y = np.max(bboxes[:, 3])
        min_x, min_y = min(min_x, x), min(min_y, y)
        max_x, max_y = max(max_x, x+wd-1), max(max_y, y+hd-1)

        img_d = img_d[min_y:max_y + 1, min_x:max_x + 1, :]
        bboxes[:, 0] -= min_x
        bboxes[:, 2] -= min_x
        bboxes[:, 1] -= min_y
        bboxes[:, 3] -= min_y

        r_w, r_h = max_x - min_x + 1, max_y - min_y + 1
        if r_w > wd or r_h > hd:
            img_d, bboxes = self._rescale2dest(img_d, np.array([r_w, r_h]), bboxes)

        return img_d.astype(np.uint8), self._dest_size, bboxes

    def _rescale2dest(self, img, size, bboxes):
        w, h = size[0], size[1]
        wd, hd = self._dest_size[0], self._dest_size[1]

        # Scale with ratio kept.
        # img_d = trans.resize(image=img, output_shape=([hd, wd]), preserve_range=True)
        img_d = cv2.resize(img, (wd, hd))
        # plt.imshow(img_d)
        # plt.draw()
        # plt.waitforbuttonpress()

        # Scale bboxes
        w_s, h_s = wd / w, hd / h
        bboxes[:, 0] = (bboxes[:, 0] * w_s).astype(np.int)
        bboxes[:, 2] = (bboxes[:, 2] * w_s).astype(np.int)
        bboxes[:, 1] = (bboxes[:, 1] * h_s).astype(np.int)
        bboxes[:, 3] = (bboxes[:, 3] * h_s).astype(np.int)

        return img_d, bboxes

def createResizePreprocessor(gconf):
    """
    Create a dataset preprocessor according to gconf.
    :param gconf: configuration, should be {'dest_size':[224, 224]}
    :return: a callable object.
    """
    return ResizePreprocessor(gconf)

def findLastCkpt(model_path):
    """
    Find last saved model in model path.
    :param model_path: Path to saved model
    :return: last ckpt file prefix and corresponding epoch number.
    """

    ckpt_prefix, epoch_num = '', 0
    ckpt_files = glob.glob(path.join(model_path, '*.ckpt.meta'))
    ckpt_files.sort()
    if len(ckpt_files) > 0:
        last = ckpt_files[-1]
        ckpt_prefix = last[last.rfind('/')+1:-5]
        epoch_num = int(ckpt_prefix[-8:-5]) + 1

    return ckpt_prefix, epoch_num

if __name__ == '__main__':
    ckpt_pref, epoch_num = findLastCkpt('../models')
    print(ckpt_pref, epoch_num)
