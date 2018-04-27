"""
MobileNet implementation.
"""
import network.featureExtractorBase as ftBase
import tensorflow as tf
import tensorlayer as tl

class MobileNet(ftBase.FeatureExtractorBase):
    """
    MobileNet implementation.
    """
    def __init__(self, conf):
        self._conf = conf
        self._inited = False
        pass

    def _depthwiseLayer(self, pre_layer, scope_name, shape):
        with tf.variable_scope(scope_name):
            pre_layer = tl.layers.DepthwiseConv2d(pre_layer, shape=(3, 3), name='conv1')
            pre_layer = tl.layers.BatchNormLayer(pre_layer, act=tf.nn.relu, is_train=self._conf['is_train'], name='bn1')
            pre_layer = tl.layers.Conv2dLayer(pre_layer, shape=shape, strides=(1,2,2,1), name='conv2')
            pre_layer = tl.layers.BatchNormLayer(pre_layer, act=tf.nn.relu, is_train=self._conf['is_train'], name='bn2')

        return pre_layer

    def predict(self, input_img):
        with tf.variable_scope('MobileNet'):
            if self._conf['resolution_mult'] == 1:
                # TODO Resize input
                pass

            net = tl.layers.InputLayer(input_img)
            with tf.variable_scope('layer-1'):
                net = tl.layers.Conv2dLayer(net, shape=(3,3,3,32), strides=(1,2,2,1))
                net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=self._conf['is_train'])

            net = self._depthwiseLayer(net, 'layer-2', (1, 1, 32, 64))
            net = self._depthwiseLayer(net, 'layer-3', (1, 1, 64, 128))
            net = self._depthwiseLayer(net, 'layer-4', (1, 1,128, 128))
            net = self._depthwiseLayer(net, 'layer-5', (1, 1, 128, 256))
            net = self._depthwiseLayer(net, 'layer-6', (1, 1, 256, 512))

            for i in range(5):
                net = self._depthwiseLayer(net, 'layer-%d' % (i + 7), (1, 1, 512, 512))
                a = 'a'

            net = self._depthwiseLayer(net, 'layer-12', (1, 1, 512, 1024))
            net = self._depthwiseLayer(net, 'layer-13', (1, 1, 1024, 1024))


            with tf.variable_scope('avgpool-layer-14'):
                net = tl.layers.PoolLayer(net, (1, 7, 7, 1), strides=(1,1,1,1), pool=tf.nn.avg_pool)

            # size = (tf.shape(net.outputs)[0], tf.shape(net.outputs)[1], tf.shape(net.outputs)[2],
            #       tf.shape(net.outputs)[3])
            with tf.variable_scope('fc-layer-15'):
                net = tl.layers.FlattenLayer(net)
                net = tl.layers.DenseLayer(net, self._conf['class_num'], act=tf.nn.relu)

        return net.outputs

