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

        ft, mbnet_layers = self._feature_extractor.predict(input_img)

        # Get specific layer output
        last_layer = mbnet_layers['layer-12'] # shape 14x14x512

        # add auxiliary conv layers to the end of backbone network
        endpoints = mbnet_layers

        with tf.variable_scope('SSDNet'):
            block='block-1'
            with tf.variable_scope(block):#14x14x512 -> 7x7x512
                net = tl.layers.Conv2dLayer(last_layer, act=tf.nn.relu, shape=(1, 1, 512, 256), name='conv3')
                net = tl.layers.Conv2dLayer(net, act=tf.nn.relu, shape=(3, 3, 256, 512), strides=(1,2,2,1), name='conv4')
            endpoints[block] = net

            block = 'block-2'
            with tf.variable_scope(block):#7x7x512 -> 4x4x256
                net = tl.layers.Conv2dLayer(net, act=tf.nn.relu, shape=(1, 1, 512, 128), name='conv5')
                net = tl.layers.Conv2dLayer(net, act=tf.nn.relu, shape=(3, 3, 128, 256), strides=(1, 2, 2, 1), name='conv6')
            endpoints[block] = net

            block = 'block-3'
            with tf.variable_scope(block):#4x4x256 -> 2x2x256
                net = tl.layers.Conv2dLayer(net, act=tf.nn.relu, shape=(1, 1, 256, 128), name='conv7')
                net = tl.layers.Conv2dLayer(net, act=tf.nn.relu, shape=(3, 3, 128, 256), strides=(1, 2, 2, 1), name='conv8')
            endpoints[block] = net

            block = 'block-4'
            with tf.variable_scope(block):#2x2x256 -> 1x1x256
                net = tl.layers.Conv2dLayer(net, act=tf.nn.relu, shape=(1, 1, 256, 128), name='conv9')
                net = tl.layers.Conv2dLayer(net, act=tf.nn.relu, shape=(2, 2, 128, 256), padding='VALID', name='conv10')
            endpoints[block] = net

        """
        Add classifier conv layers to each added feature map(including the last layer of backbone network).
        Prediction and localisations layers.
        """
        predictions = []
        logits = []
        locations = []
        for layer in self._net_conf['featuremaps'].keys():
            with tf.variable_scope(layer + '_box'):
                prob, loc = self._multibox_predict(endpoints[layer],
                                                   self._net_conf['class_num'],
                                                   self._net_conf['featuremaps'][layer])

                predictions.append(tf.nn.softmax(prob))
                logits.append(prob)
                locations.append(loc)

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


    def _multibox_predict(self, input, class_num, layer_conf):
        """
        Compute predictions for each input layer.
        :param input_layer: Input feature layer
        :param class_num: number of output classes.
        :param anchor_num: number of anchors.
        :return: prediction p, and localizatoin l.
        """
        anchor_num = len(layer_conf.ratios) + 1
        input_shape = (int(input.outputs.get_shape()[1]),
                       int(input.outputs.get_shape()[2]),
                       int(input.outputs.get_shape()[3]))

        with tf.variable_scope('pred'):
            pred_rst = tl.layers.Conv2dLayer(input, shape=(3, 3, input_shape[2], anchor_num * (class_num + 4)))
            pred_rst = tl.layers.ReshapeLayer(pred_rst, shape=(1, input_shape[0], input_shape[1], anchor_num, class_num + 4))

        # Reshape output tensor to extract each anchor prediction.
        pred_class = tf.slice(pred_rst.outputs, [0, 0, 0, 0, 0], [self._net_conf['batch_size'],
                                                                  input_shape[0],
                                                                  input_shape[1],
                                                                  anchor_num,
                                                                  class_num])

        pred_pos = tf.slice(pred_rst.outputs, [0, 0, 0, 0, class_num], [self._net_conf['batch_size'],
                                                                        input_shape[0],
                                                                        input_shape[1],
                                                                        anchor_num,
                                                                        4])
        print(pred_pos.get_shape())
        return pred_class, pred_pos


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