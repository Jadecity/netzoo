import tensorflow as tf
import tensorlayer as tl

if __name__ == '__main__':
    a = tf.placeholder(tf.int8, [2, 3])
    b = tf.placeholder(tf.int8, [None, None, 3])
    c = tf.reshape(a, [tf.div(tf.shape(a)[0], 3), 3])
    d = []
    # i = 1
    # while tf.less(i, tf.shape(a)[0]):
    #     i = i + 1
    d = tf.slice(a, [0,0], [2, 2])

    with tf.Session() as ss:
        d = ss.run((d), feed_dict={a: [[1,2,3],[4,5,6]]})

        print(d)
    # last_layer = tf.placeholder(tf.float32, shape=(1,224,224,3))
    # input = tl.layers.InputLayer(last_layer)
    # net = tl.layers.Conv2dLayer(input, act=tf.nn.relu, shape=(3, 3, 3, 1024), name='conv1')
    #
    # print(tl.layers.get_layers_with_name(net, 'conv1'))
    # print(net.all_layers)
    # print(net.outputs)