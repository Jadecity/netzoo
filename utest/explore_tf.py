import tensorflow as tf
import tensorlayer as tl
from collections import namedtuple
import common.utils as utils
import matplotlib.pyplot as pl

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

    # last_layer = tf.placeholder(tf.float32, shape=(1,7,7,5))
    # input = tl.layers.InputLayer(last_layer)
    # net = tl.layers.PoolLayer(input, ksize=(1, 7, 7, 1), strides=(1,1,1,1), name='pool1', padding='VALID')
    # shape = tf.shape(net.outputs)
    #
    # with tf.Session() as ss:
    #     shape = ss.run(shape, feed_dict={last_layer:np.ones([1,7,7,5])})
    #
    # print(shape)

    # FeatmapConf = namedtuple('tp', ['name', 'size'])
    # a = FeatmapConf(name='ft', size=1)
    # print(a.size)

    # a = tf.Variable([1,2,3], dtype=tf.int64)
    # a = tf.one_hot([[1],[2],[3]], 5, dtype=tf.int64)
    #
    # print(a.get_shape())
    # ss = tf.InteractiveSession()
    # print(ss.run(a))
    # ss.close()

    # a = tf.constant([[1,2,3]])
    # b = tf.split(a, 3, axis=1)
    # c = b[0]
    # # c = tf.minimum(b[0], tf.slice(a, [0], [1]))
    # ss = tf.InteractiveSession()
    # c = ss.run(c)
    # print(c)

    # block = np.zeros([2,2,3,4])
    # b1 = tf.constant([1,1,3,3], dtype=tf.float32)
    # b2 = tf.constant([[[[1,1,3,3],
    #                   [3,3,3,3]]]], dtype=tf.float32)
    #
    # area = utils.jaccardIndex(b1, b2)
    # ss = tf.InteractiveSession()
    # area = ss.run(area)
    # print(area)
    # ss.close()

    # x = tf.constant(2)
    # y = tf.constant(5)
    #
    # r = tf.cond(tf.less(x, y), lambda :tf.multiply(x, 17), lambda :tf.subtract(tf.abs(x), 1))
    #

    # x = tf.constant(2, dtype=tf.float32)
    # fx = utils.smoothL1(x)

    # a = tf.constant([[1,1,3,3], [5,2,3,3]])
    b = tf.constant([200,300,400,500])
    ss = tf.InteractiveSession()
    # # rst = ss.run(tf.logical_not( tf.greater(b, 2)))
    # # print(rst)
    # #
    # # rst = ss.run(tf.where(tf.greater(b, 2)))
    # #
    # # print(rst)
    # #
    # # rst = ss.run(tf.gather_nd(b, tf.where(tf.greater(b, 2))))
    # # print(rst)
    # #
    rst,idx = ss.run(tf.nn.top_k(b, 2))
    print(rst, idx)
    #
    # count = tf.greater(b, 2)
    # count = tf.reduce_sum(tf.cast(count, dtype=tf.int8))
    # rst = ss.run(count)
    # print(rst)


    # rst= ss.run(a[0])
    # print(rst)
    #
    # a_x = a[:, 0]
    # b_x = b[0]
    # c = tf.subtract(a_x, b_x)
    # rst = ss.run(c)
    # print(rst)

    # a = tf.constant([0.1, 0.2, 0.3, 0.4])
    # a = tf.expand_dims(a, 0)
    # a = tf.expand_dims(a, 0)
    # b = tf.tile(a, [3,3,1])
    # # b = a[0]
    # # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=b, logits=a)
    # ss = tf.InteractiveSession()
    # loss = ss.run(b)
    # print(loss)

    # a = tf.constant([[0.1, 0.2, 0.3, 0.4],
    #                  [0.1, 0.2, 0.3, 0.4]])
    # b = tf.reduce_sum(a)
    # # b = a[0]
    # # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=b, logits=a)

    # a = tf.constant([0, 1, 0.2, 1, 2, 3])
    # b = utils.smoothL1(a)

    #
    # gboxes = np.array([0.2, 0.2, 0.1, 0.2])
    # gboxes = tf.constant(gboxes)
    # gboxes = tf.reshape(gboxes, shape=[1, 4])
    # ss = tf.InteractiveSession()
    # loss = ss.run(gboxes)
    # print(loss)

    # a = 2.0
    # b = tf.constant([2] ,dtype=tf.float32)
    # d = tf.add(b, 2.0)
    # c = tf.divide(d, 2.0)
    # ss = tf.InteractiveSession()
    # loss = ss.run(c)
    # print(loss)
    # img = np.zeros([10, 10])
    # pl.imshow(img)
    # print(pl.waitforbuttonpress())
    # pl.close()

    # a = np.array([0,1,0,0])
    # b = np.array([[2,3,4,5]])
    # a = tf.constant(a, dtype=tf.int32)
    # b = tf.constant(b, dtype=tf.int32)
    # c = tf.greater(a, 0)
    # d = tf.where(c)
    # e = tf.gather_nd(b, d)
    # # g = tf.stack([0, d[0, 0]], axis=0)
    # m = tf.concat([[0], d[0]], axis=0)
    #
    # f = tf.slice(b, m, [1, 1])
    # g = tf.nn.top_k(a, 3)
    #
    # ss = tf.InteractiveSession()
    # loss = ss.run(3*a)
    # print(loss)


    # a = tf.Variable([1,2,3], dtype=tf.float32)
    # tf.add_to_collection('my_var', a)
    # b = tf.get_collection('my_var')
    # init = tf.global_variables_initializer()
    # with tf.Session() as ss:
    #     ss.run(init)
    #     print(ss.run(b))

    # a = tf.constant(np.ones([10, 10]), dtype=tf.float32)
    # a = tf.reshape(a, (1, 10, 10, 1))
    # input = tl.layers.InputLayer(a)
    # net = tl.layers.Conv2dLayer(input, shape=(3,3,1, 10), name='conv1')
    # net = tl.layers.FlattenLayer(net)
    # net = tl.layers.DenseLayer(net, 5)
    # print(net.all_params)
    # loss = tf.constant(0, dtype=tf.float32)
    #
    # for var in tl.layers.get_variables_with_name('W_conv2d'):
    #     tf.add(loss, tf.nn.l2_loss(var))
    # print(loss.get_shape())

    # net = tl.layers.PoolLayer(net, ksize=(1,2,2,1))
    # tf.add_to_collection('weights', net.all_params)
    # all_weights = tf.add_n(tf.reduce_sum(tf.get_collection('weights')))
    # with tf.Session() as ss:
    #     print(ss.run(loss))

    # a = tf.constant(np.ones([1, 5]))
    # m = tf.constant(1, tf.int32)
    # b = tf.slice(a, [0, 0], [m, 1])
    # a = [3]
    # b = tf.constant([4,4])
    # c = tf.concat([b, a], axis=0)

    # c = tf.constant([[1,2,3], [4,5,3]], dtype=tf.float32)
    # d = tf.argmax(c, 1)
    # e = tf.nn.softmax(c)
    #
    # with tf.Session() as ss:
    #     print(ss.run(e))

    # a = [1,2,3]
    # print(np.argmax(a))

    # a = [1,2,3]
    # b = np.array(a)
    # c = b.flatten()
    # print(c)
    #
    # print(np.array([0]).flatten())
    # a = np.array([0])
    # for b in a:
    #     print(b)
    pass