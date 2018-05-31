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
        self._net = self._build_net(input)

    def get_output(self):
        return self._net.outputs

    def _build_net(self, input):
        net = ly.InputLayer(input, name='input_layer')
        with tf.variable_scope('ResNeXt29', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('layer0'):  # 32x32x3
                net = ly.Conv2d(net, 64, W_init=tf.contrib.layers.xavier_initializer(), name='Conv0')
                net = ly.BatchNormLayer(net, act=tf.nn.relu, is_train=self._conf.is_train, name='bn0')

            with tf.variable_scope('stage-1'):  # 32x32x64->32x32x256
                net = self._resNeXt_block(net, out_dim=(64, 64, 256), stride=(1, 1), card=self._conf.card)

            with tf.variable_scope('stage-2'):  # 32x32x256->16x16x256
                net = self._resNeXt_block(net, out_dim=(128, 128, 512), stride=(2, 2), card=self._conf.card)

            with tf.variable_scope('stage-3'):  # 16x16x256->8x8x256
                net = self._resNeXt_block(net, out_dim=(512, 512, 1024), stride=(2, 2), card=self._conf.card)

            with tf.variable_scope('global-avg-pooling'):
                net = myPooling.GlobalMeanPool2d(net, 'pool')
                net = tl.layers.ReshapeLayer(net, (-1, 1, 1, 1024), name='Reshape')

            with tf.variable_scope('fc-layer'):
                net = tl.layers.Conv2d(net, self._conf.class_num, (1, 1), W_init=tf.contrib.layers.xavier_initializer(),
                                       name='dense')
                net = tl.layers.FlattenLayer(net)

        return net

    def _resNeXt_block(self, input, out_dim, stride, card):
        """
        Build a resNeXt block.
        :param input: previous layer
        :param out_dim: a 3-element tuple defines each layer output dimension.
        :param stride: stride of the middle convolution layer.
        :param card: cardinality in the paper.
        :return:
        """
        branch_sum = tf.constant(0, dtype=tf.float32)
        for i in range(card):
            with tf.variable_scope('card-%d' % i):
                net = ly.Conv2d(input, out_dim[0], filter_size=(1, 1), W_init=tf.contrib.layers.xavier_initializer(),
                                name='conv1_1')
                net = ly.BatchNormLayer(net, act=tf.nn.relu, is_train=self._conf.is_train, name='bn1')

                net = ly.Conv2d(net, out_dim[1], filter_size=(3, 3), strides=stride,
                                W_init=tf.contrib.layers.xavier_initializer(), name='conv2_3')
                net = ly.BatchNormLayer(net, act=tf.nn.relu, is_train=self._conf.is_train, name='bn2')

                net = ly.Conv2d(net, out_dim[2], filter_size=(1, 1), W_init=tf.contrib.layers.xavier_initializer(),
                                name='conv3_1')
                net = ly.BatchNormLayer(net, act=tf.nn.relu, is_train=self._conf.is_train, name='bn3')

                branch_sum = tf.add(branch_sum, net.outputs)

        with tf.variable_scope('shortcut'):
            input_val = input
            if input.outputs.get_shape()[-1].value != out_dim[2]:
                input_val = ly.Conv2d(input_val, out_dim[2], filter_size=(1, 1), strides=stride, padding='valid',
                                      W_init=tf.contrib.layers.xavier_initializer(), name='conv')
                input_val = ly.BatchNormLayer(input_val, is_train=self._conf.is_train, name='bn')

        branch_sum = tf.add(branch_sum, input_val.outputs)
        branch_sum = tf.nn.relu(branch_sum)
        net = ly.InputLayer(branch_sum)

        return net