import pickle
import glob
import os.path as path
import numpy as np
import common.utils as utils
import matplotlib.pyplot as plt

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

def cifar_to_annotation_json(cifar_home, dest_json_path):
    """
    Convert standard cifar data set to standard annotation json files.
    :return: None
    """
    label_names = unpickle(path.join(cifar_home, 'batches.meta'))
    label_names = label_names[b'label_names']

    for fname in glob.glob(path.join(cifar_home, 'data_batch_*')):
        data_batch = unpickle(fname)
        cnt = 1
        for label, raw_img, filename in zip(data_batch[b'labels'], data_batch[b'data'], data_batch[b'filenames']):
            r = np.reshape(raw_img[0:32**2], [32, 32])
            g = np.reshape(raw_img[32**2:2*(32**2)], [32, 32])
            b = np.reshape(raw_img[2*(32**2):3*(32**2)], [32, 32])

            img = np.dstack((r, g, b))

            utils.visulizeClassV2(img, label, label_names, hold=True)
            plt.waitforbuttonpress()

if __name__ == '__main__':
    cifar_home = '/home/autel/data/cifar/cifar-10-batches-py/'
    cifar_to_annotation_json(cifar_home, '')