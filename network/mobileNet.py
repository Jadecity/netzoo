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
        self._all_layers = {}
        pass

    def _depthwiseLayer(self, pre_layer, scope_name, shape, strides=(1, 1)):
        with tf.variable_scope(scope_name):
            pre_layer = tl.layers.DepthwiseConv2d(pre_layer, shape=(3, 3), strides=strides, b_init=None, name='conv1')
            pre_layer = tl.layers.BatchNormLayer(pre_layer, act=tf.nn.relu6, is_train=self._conf['is_train'], name='bn1')
            pre_layer = tl.layers.Conv2dLayer(pre_layer, shape=shape, b_init=None, name='conv2')
            pre_layer = tl.layers.BatchNormLayer(pre_layer, act=tf.nn.relu6, is_train=self._conf['is_train'], name='bn2')

        return pre_layer

    def predict(self, input_img):
        with tf.variable_scope('MobileNet'):
            if self._conf['resolution_mult'] == 1:
                # TODO Resize input
                pass

            net = tl.layers.InputLayer(input_img)
            with tf.variable_scope('layer-0'):  #input 224x224x3
                net = tl.layers.Conv2dLayer(net, shape=(3,3,3,32), strides=(1,2,2,1), b_init=None)
                net = tl.layers.BatchNormLayer(net, act=tf.nn.relu6, is_train=self._conf['is_train'])
                self._all_layers['layer-0'] = net

            net = self._depthwiseLayer(net, 'layer-1', (1, 1, 32, 64))                  #112x112x32 ->112x112x64
            self._all_layers['layer-1'] = net

            net = self._depthwiseLayer(net, 'layer-2', (1, 1, 64, 128), strides=(2, 2)) #112x112x64 ->56x56x128
            self._all_layers['layer-2'] = net

            net = self._depthwiseLayer(net, 'layer-3', (1, 1,128, 128))                 #56x56x128 ->56x56x128
            self._all_layers['layer-3'] = net

            net = self._depthwiseLayer(net, 'layer-4', (1, 1, 128, 256), strides=(2, 2))#56x56x128 ->28x28x256
            self._all_layers['layer-4'] = net

            net = self._depthwiseLayer(net, 'layer-5', (1, 1, 256, 256))                #28x28x256 ->28x28x256
            self._all_layers['layer-5'] = net

            net = self._depthwiseLayer(net, 'layer-6', (1, 1, 256, 512), strides=(2, 2))#28x28x256 ->14x14x512
            self._all_layers['layer-6'] = net

            net = self._depthwiseLayer(net, 'layer-7', (1, 1, 512, 512))#14x14x512 ->14x14x512
            self._all_layers['layer-7'] = net
            net = self._depthwiseLayer(net, 'layer-8', (1, 1, 512, 512))#14x14x512 ->14x14x512
            self._all_layers['layer-8'] = net
            net = self._depthwiseLayer(net, 'layer-9', (1, 1, 512, 512))#14x14x512 ->14x14x512
            self._all_layers['layer-9'] = net
            net = self._depthwiseLayer(net, 'layer-10', (1, 1, 512, 512))#14x14x512 ->14x14x512
            self._all_layers['layer-10'] = net
            net = self._depthwiseLayer(net, 'layer-11', (1, 1, 512, 512))#14x14x512 ->14x14x512
            self._all_layers['layer-11'] = net

            net = self._depthwiseLayer(net, 'layer-12', (1, 1, 512, 1024), strides=(2, 2)) #14x14x512 -> 7x7x1024
            self._all_layers['layer-12'] = net

            net = self._depthwiseLayer(net, 'layer-13', (1, 1, 1024, 1024))#7x7x1024 -> 7x7x1024
            self._all_layers['layer-13'] = net


            with tf.variable_scope('avgpool-layer-15'):
                net = GlobalMeanPool2d(net, 'pool')
                net = tl.layers.ReshapeLayer(net, (-1, 1, 1, 1024), name='Reshape')
                self._all_layers['avgpool-layer-15'] = net

            print(net.outputs.get_shape())

            # size = (tf.shape(net.outputs)[0], tf.shape(net.outputs)[1], tf.shape(net.outputs)[2],
            #       tf.shape(net.outputs)[3])
            with tf.variable_scope('fc-layer-16'):
                net = tl.layers.Conv2d(net, self._conf['class_num'], (1, 1), (1, 1), name='dense')
                net = tl.layers.FlattenLayer(net)
                self._all_layers['fc-layer-16'] = net

        return net, self._all_layers

    def _mean_pool(self, pre_layer, name):
        tf.reduce_mean(pre_layer.outputs, axis=[1, 2], name=name)


class GlobalMeanPool2d(tl.layers.Layer):
    """The :class:`GlobalMeanPool2d` class is a 2D Global Mean Pooling layer.

    Parameters
    ------------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 4 [batch, height, width, channel].
    name : str
        A unique layer name.

    Examples
    ---------
    >>> x = tf.placeholder("float32", [None, 100, 100, 30])
    >>> n = InputLayer(x, name='in2')
    >>> n = GlobalMeanPool2d(n)
    ... [None, 30]
    """

    def __init__(self, prev_layer, name='globalmeanpool2d'):
        super(GlobalMeanPool2d, self).__init__(prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs
        # operation (customized)
        self.outputs = tf.reduce_mean(self.inputs, axis=[1, 2], name=name)
        # update layer (customized)
        self.all_layers.append(self.outputs)