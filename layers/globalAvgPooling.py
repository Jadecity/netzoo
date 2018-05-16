import tensorflow as tf
import tensorlayer as tl


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