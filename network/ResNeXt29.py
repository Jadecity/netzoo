"""
Implement ResNeXt29 for cifar data set.
"""
import tensorlayer.layers as ly
import tensorflow as tf
import tensorlayer as tl
import layers.globalAvgPooling as myPooling

class ResNeXt29:

    def __init__(self, conf, input):
        self._conf = conf
        self._output =  self._build_net(input)

    def get_output(self):
        return self._output

    def _build_net(self, input):
        net = ly.InputLayer(input, name='input_layer')

        with tf.name_scope('layer0'): # 32x32x3
            net = ly.Conv2d(net, 64, name='Conv0')
            net = ly.BatchNormLayer(net, act=tf.nn.relu6, name='bn0')

        with tf.name_scope('stage-1'):# 32x32x64->32x32x256
            net = self._resNeXt_block(net, out_dim=(64, 64, 256), last_stride=(1, 1), card=self._conf['card'])

        with tf.name_scope('stage-2'):# 32x32x256->16x16x256
            net = self._resNeXt_block(net, out_dim=(128, 128, 512), last_stride=(2, 2), card=self._conf['card'])

        with tf.name_scope('stage-3'):# 16x16x256->8x8x256
            net = self._resNeXt_block(net, out_dim=(512, 512, 1024), last_stride=(2, 2), card=self._conf['card'])

        with tf.name_scope('global-avg-pooling'):
            net = myPooling.GlobalMeanPool2d(net, 'pool')
            net = tl.layers.ReshapeLayer(net, (-1, 1, 1, 1024), name='Reshape')

        with tf.variable_scope('fc-layer'):
            net = tl.layers.Conv2d(net, self._conf['class_num'], (1, 1), name='dense')
            net = tl.layers.FlattenLayer(net)

        self._net = net

    def _resNeXt_block(self, input, out_dim, last_stride, card):
        """
        Build a resNeXt block.
        :param input: previous layer
        :param out_dim: a 3-element tuple defines each layer output dimension.
        :param last_stride: stride of the last convolution layer.
        :param card: cardinality in the paper.
        :return:
        """
        branch_sum = tf.Constant(0)
        for i in range(card):
            with tf.name_scope('card-%d' % i):
                net = ly.Conv2d(input, out_dim[0], filter_size=(1, 1), name='conv1_1')
                net = ly.BatchNormLayer(net, act=tf.nn.relu6, name='bn1')

                net = ly.Conv2d(net, out_dim[1], filter_size=(3, 3), name='conv2_3')
                net = ly.BatchNormLayer(net, act=tf.nn.relu6, name='bn2')

                net = ly.Conv2d(net, out_dim[3], filter_size=(1, 1), strides=last_stride, name='conv3_1')
                net = ly.BatchNormLayer(net, act=tf.nn.relu6, name='bn3')

                branch_sum = tf.add(branch_sum, net.outputs)

        branch_sum = tf.add(branch_sum, input.output)
        net = tf.nn.relu6(branch_sum)

        return ly.InputLayer(net)

