"""
Reimplementation of SSD[1]
[1]: SSD: Single Shot MultiBox Detector
"""
import tensorlayer as tl
import tensorflow as tf
import math
import numpy as np

class SSDNet:
    def __init__(self, net_conf, feature_extractor):
        """
        Build a SSD network using a backbone feature extractor.
        :param net_conf: SSD network configuration.
        :param feature_extractor: backbone feature extractor.
        :return None
        """
        self._feature_extractor = feature_extractor
        self._net_conf = net_conf

    def predict(self, input_img, is_training=False):
        """
        Predict labels and bboxes in image.
        :param input_img: input image.
        :param is_training: True during training period, otherwise false.
        :return: labels , corresponding bboxes, confidential and endpoints of the network.
        """

        ft = self._feature_extractor.predict(input_img)

        # Get specific layer output
        last_layer = tl.layers.get_layers_with_name(ft, 'layer-6')[-1]
        last_layer = tl.layers.InputLayer(last_layer)

        # add auxiliary conv layers to the end of backbone network
        endpoints = {}
        endpoints['block-0'] = last_layer

        with tf.variable_scope('SSDNet'):
            block = 'block-1'
            with tf.variable_scope(block):
                net = tl.layers.Conv2dLayer(last_layer, act=tf.nn.relu, shape=(3, 3, 256, 1024), name='conv1')
                net = tl.layers.Conv2dLayer(net, act=tf.nn.relu, shape=(1, 1, 1024, 1024), name='conv2')
            endpoints[block] = net

            block='block-2'
            with tf.variable_scope(block):
                net = tl.layers.Conv2dLayer(net, act=tf.nn.relu, shape=(1, 1, 1024, 256), name='conv3')
                net = tl.layers.Conv2dLayer(net, act=tf.nn.relu, shape=(3, 3, 256, 512), strides=(1,2,2,1), name='conv4')
            endpoints[block] = net

            block = 'block-3'
            with tf.variable_scope(block):
                net = tl.layers.Conv2dLayer(net, act=tf.nn.relu, shape=(1, 1, 512, 128), name='conv5')
                net = tl.layers.Conv2dLayer(net, act=tf.nn.relu, shape=(3, 3, 128, 256), strides=(1, 2, 2, 1), name='conv6')
            endpoints[block] = net

            block = 'block-4'
            with tf.variable_scope(block):
                net = tl.layers.Conv2dLayer(net, act=tf.nn.relu, shape=(1, 1, 256, 128), name='conv7')
                net = tl.layers.Conv2dLayer(net, act=tf.nn.relu, shape=(3, 3, 128, 256), name='conv8')
            endpoints[block] = net

        """ 
        Add classifier conv layers to each added feature map(including the last layer of backbone network).
        Prediction and localisations layers.
        """
        predictions = []
        logits = []
        locations = []
        anchor_sizes, anchor_ratios, normalizations = self._get_anchors(self._net_conf['featuremap_sizes'])

        for i, layer in enumerate(endpoints):
            with tf.variable_scope(layer + '_box'):
                p, l = self._multibox_predict(endpoints[layer],
                                          self._net_conf['class_num'], len(anchor_sizes))

                predictions.append(tf.nn.softmax(p))
                logits.append(p)
                locations.append(l)

        return predictions, logits, locations, endpoints

    def _get_anchors(self, feature_map_sizes):
        """
        TODO To understand the logic that how anchors are used, and refine this function.
        Create predefined anchors according to feature map number.
        :param feature_map_sizes: Size of each square feature map, only side length.
        :return:
            anchor_sizes: Anchor size list.
            anchor_ratios: Anchor ratio list.
            normalizations: Normalized anchor position.
        """
        ratios = [1, 2, 3, 1/2.0, 1/3.0]
        sqrt_ratios = [math.sqrt(x) for x in ratios]

        s_min = 0.2
        s_max = 0.9
        sizes = []
        feature_map_num = len(feature_map_sizes)
        for k in range(feature_map_num):
            k = k + 1
            s_k = s_min + (s_max - s_min)/(feature_map_num - 1)*(k - 1)
            for ratio in math.sqrt_ratios:
                width = s_k * ratio
                height = s_k / ratio
                sizes.append([width, height])

            # Add one more for ratio 1.
            s_k_2 = s_min + (s_max - s_min)/(feature_map_num - 1)*(k + 1 - 1)
            width = math.sqrt(s_k * s_k_2)
            height = width
            sizes.append([width, height])

        norms = []
        for size in feature_map_sizes:
            norm = np.zeros([size, size, 2])
            for i in range(1, size + 1):
                for j in range(1, size + 1):
                    norm[i, j, :] = np.array((i+0.5)/size**2, (j+0.5)/size**2)

            norms.append(norm)

        return sizes, ratios, norms


    def _multibox_predict(self, input, class_num, anchor_num):
        """
        Compute predictions for each output layer.
        :param input_layer: Input feature layer of size anchor_num * (class number + 4 offsets)
        :param class_num: number of output classes.
        :param anchor_num: number of anchors.
        :return: prediction p, and localizatoin l.
        """
        with tf.variable_scope('class-pred'):
            class_end = tl.layers.Conv2dLayer(input, shape=(3,3,tf.shape(input)[3], anchor_num * class_num))

        with tf.variable_scope('pos-pred'):
            pos_end = tl.layers.Conv2dLayer(input, shape=(3,3,tf.shape(input)[3], anchor_num * 4))

        # Reshape output tensor to extract each anchor prediction.
        p = tf.reshape(class_end, shape=(anchor_num, class_num))
        l = tf.reshape(pos_end, shape=(anchor_num, 4))

        return p, l



    def loss(self, labels, bboxes, glabels, gbboxes):
        """
        Compute loss and put them to tf.collections.LOSS and other loss
        :param labels: predicted labels.
        :param bboxes: predicted bounding boxes.
        :param glabels: ground truth labels.
        :param gbboxes: gound truth bounding boxes.
        :return: None
        """

        pass