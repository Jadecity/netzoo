import tensorflow as tf
import tensorlayer as tl
from collections import namedtuple
import common.utils as utils


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
    # # b2 = tf.constant([[2,2,3,3],
    # #                   [3,3,3,3]], dtype=tf.float32)
    # b2 = tf.constant(block, dtype=tf.float32)
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
    # b = tf.constant([2,2,3,3])
    #
    # ss = tf.InteractiveSession()
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
    # # rst = ss.run(tf.nn.top_k(b, 2))
    # # print(rst[0])
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


    gboxes = np.array([0.2, 0.2, 0.1, 0.2])
    gboxes = tf.constant(gboxes)
    gboxes = tf.reshape(gboxes, shape=[1, 4])
    ss = tf.InteractiveSession()
    loss = ss.run(gboxes)
    print(loss)
