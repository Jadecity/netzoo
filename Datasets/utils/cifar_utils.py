import pickle
import glob
import os.path as path
import numpy as np
import common.utils as utils
import matplotlib.pyplot as plt
import os
import json
import cv2
import tensorflow as tf
import common.utils as utils

def unpickle(file):
    """
    Unpack a cifar-10 data file
    :param file: data batch filename.
    :return:

    data -- a 10000x3072 numpy array of uint8s.
    Each row of the array stores a 32x32 colour image.
    The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
    The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.

    labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def cifar_to_class_json(cifar_home, dest_path, write_img=False, mode='train'):
    """
    Convert standard cifar data set to standard class json files.
    :param cifar_home, path to cifar data.
    :param dest_path, destination path that image data and label file will produce.
    :param write_img, unpack images if True.
    :return: None
    """
    label_names = unpickle(path.join(cifar_home, 'batches.meta'))
    label_names = label_names[b'label_names']

    # make dest path for images, json files
    img_dir = path.join(dest_path, mode, 'imgs')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    json_file = open(path.join(dest_path, 'img_labels.json'), 'w')
    if mode == 'train':
        file_pattern = 'data_batch_*'
    elif mode == 'eval':
        file_pattern = 'test_batch'

    name_label = {}
    for fname in glob.glob(path.join(cifar_home, file_pattern)):
        data_batch = unpickle(fname)
        for label, raw_img, filename in zip(data_batch[b'labels'], data_batch[b'data'], data_batch[b'filenames']):
            filename = str(filename, encoding='utf8')
            name_label[filename] = {
                'image_name': filename,
                'width': 32,
                'height': 32,
                'channel': 3,
                'label': label,
                'label_name': str(label_names[label], encoding='utf8')
            }

            if write_img:
                r = np.reshape(raw_img[0:32**2], [32, 32])
                g = np.reshape(raw_img[32**2:2*(32**2)], [32, 32])
                b = np.reshape(raw_img[2*(32**2):3*(32**2)], [32, 32])

                img = np.dstack((r, g, b))
                cv2.imwrite(path.join(img_dir, filename), img)

    json.dump(name_label, json_file)
    json_file.close()

class CifarPreprocessor:
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

def cifar_preprocessor(conf):
    return CifarPreprocessor(conf)

def encode2Tfrecord(src_path, data_home, dest_path, cnt_max, preprocessor=None):
    """
     Read json files in src_path, read corresponding image file,
     and convert them to tfrecord format.
     :param src_path: directory which contains json files.
     :param data_home: home prefix to image name in json files.
     :param dest_path: destination directory where to put tfrecords.
     :param cnt_max: max number of tfrecord in each file.
     :param preprocessor, will preprocess inputs.
     :return: none.
     """

    if not path.exists(dest_path):
        os.makedirs(dest_path)

    cnt = 1

    tfrecord_filename = os.path.join(dest_path, '%d.tfrecords' % (cnt))
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)

    content = open(os.path.join(src_path, 'img_labels.json')).read()
    ori_rcd = json.loads(content)

    mean_img = np.zeros([32, 32, 3], dtype=np.float32)
    for img_name in ori_rcd.keys():
        img = cv2.imread(path.join(data_home, img_name))
        img_size = [ori_rcd[img_name]['width'],
                    ori_rcd[img_name]['height'],
                    ori_rcd[img_name]['channel']]
        img_size = np.array(img_size)
        label = ori_rcd[img_name]['label']
        label_name = ori_rcd[img_name]['label_name']

        mean_img = np.add(mean_img, img)

        # utils.visulizeClassByName(img, ori_rcd[img_name]['label_name'], hold=True)
        # plt.waitforbuttonpress()

        if preprocessor != None:
            img, img_size = preprocessor(img, img_size)

        feature = {
            'image_name': utils.bytes_feature(tf.compat.as_bytes(img_name)),
            'image': utils.bytes_feature(img.tobytes()),
            'size': utils.int64List_feature(img_size),
            'label': utils.int64_feature(label),
            'label_name': utils.bytes_feature(tf.compat.as_bytes(label_name))
        }

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Writing the serialized example.
        example_str = example.SerializeToString()
        writer.write(example_str)

        print('cnt: %d' % cnt)
        cnt = cnt + 1

        # write out records each cnt_max items
        if 0 == cnt % cnt_max:
            writer.close()
            tfrecord_filename = os.path.join(dest_path, '%d.tfrecords' % (cnt))
            writer = tf.python_io.TFRecordWriter(tfrecord_filename)

        # break
    writer.close()

    # write mean image to file.
    mean_img /= cnt
    np.save(path.join(cifar_home, 'mean_img.npy'), mean_img)


if __name__ == '__main__':
    cifar_home = '/home/autel/data/cifar/cifar-10-batches-py/'
    tfrecord_path = '/home/autel/data/cifar/cifar-10-batches-py/tfrecords/eval'
    data_home = path.join(cifar_home, 'eval/imgs')

    # cifar_to_class_json(cifar_home, cifar_home, write_img=True, mode='eval')
    encode2Tfrecord(cifar_home, data_home=data_home, dest_path=tfrecord_path, cnt_max=1000)