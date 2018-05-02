import tensorflow as tf
import tensorlayer as tl
from collections import namedtuple



import numpy as np

if __name__ == '__main__':
    # a = tf.placeholder(tf.int8, [2, 3])
    # b = tf.placeholder(tf.int8, [None, None, 3])
    # c = tf.reshape(a, [tf.div(tf.shape(a)[0], 3), 3])
    # d = []
    # # i = 1
    # # while tf.less(i, tf.shape(a)[0]):
    # #     i = i + 1
    # d = tf.slice(a, [0,0], [2, 2])
    #
    # with tf.Session() as ss:
    #     d = ss.run((d), feed_dict={a: [[1,2,3],[4,5,6]]})
    #
    #     print(d)

    tl.layers.Conv2d
    # last_layer = tf.placeholder(tf.float32, shape=(1,7,7,5))
    # input = tl.layers.InputLayer(last_layer)
    # net = tl.layers.PoolLayer(input, ksize=(1, 7, 7, 1), strides=(1,1,1,1), name='pool1', padding='VALID')
    # shape = tf.shape(net.outputs)
    #
    # with tf.Session() as ss:
    #     shape = ss.run(shape, feed_dict={last_layer:np.ones([1,7,7,5])})
    #
    # print(shape)

    FeatmapConf = namedtuple('tp', ['name', 'size'])
    a = FeatmapConf(name='ft', size=1)
    print(a.size)