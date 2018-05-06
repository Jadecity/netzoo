"""
Reimplementation of SSD[1]
[1]: SSD: Single Shot MultiBox Detector
"""
import tensorlayer as tl
import tensorflow as tf
import math
import numpy as np
import common.utils as utils

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
        self._anchors = self._create_anchors(self._net_conf['featuremaps'])

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
        for layer_conf in self._net_conf['featuremaps']:
            with tf.variable_scope(layer_conf.layer_name + '_box'):
                prob, loc = self._multibox_predict(endpoints[layer_conf.layer_name],
                                                   self._net_conf['class_num'],
                                                   layer_conf)

                predictions.append(tf.nn.softmax(prob))
                logits.append(prob)
                locations.append(loc)

        return predictions, logits, locations, endpoints

    def _create_anchors(self, fm_conf):
        """
        Create predefined anchors according to feature map number.
        :param feature_map_sizes: Size of each square feature map, only side length.
        :return:
            anchor_sizes: Anchor size list.
            anchor_ratios: Anchor ratio list.
            normalizations: Normalized anchor position.
        """

        boxes_list = []
        for i, layer_conf in enumerate(fm_conf):
            sqrt_ratios = [math.sqrt(x) for x in layer_conf.ratios]
            boxes = []

            sk = layer_conf.scale
            for ratio in sqrt_ratios:
                width = sk * ratio
                height = sk / ratio
                boxes.append([width, height])

            # Add one more for ratio 1.
            if i < len(fm_conf) - 1:
                s_prime = math.sqrt(sk * fm_conf[i + 1].scale)
            else:
                s_prime = math.sqrt(sk * 107.5)

            boxes.append([s_prime, s_prime])
            boxes_list.append(boxes)

        # compute anchors for each position
        layer_anchors = {}
        for i, layer_conf in enumerate(fm_conf):
            anchors = np.zeros([layer_conf.size, layer_conf.size, len(layer_conf.ratios) + 1, 4])
            for r in range(layer_conf.size):
                cy = (r+0.5)/layer_conf.size**2
                for c in range(layer_conf.size):
                    cx = (c+0.5)/layer_conf.size**2
                    for k, boxes in enumerate(boxes_list[i]):
                        anchors[r, c, k, :] = [cx, cy, boxes[0], boxes[1]]

            # save anchors for each feature map
            layer_anchors[layer_conf.layer_name] = anchors

        return layer_anchors

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

    def _confLoss(self, pos_mask, neg_mask, logits):
        """
        Compute confidence loss.
        :param pos_mask: positive example boolean mask.
        :param neg_mask: negative example boolean mask.
        :param glabel: ground truth label, shape[class_num]
        :param logits: predicted probability, shape[height, width, anchor_num, class_num]
        :return:
        """

        # Loss for each postition.
        conf_loss = tf.nn.softmax(logits, axis=3)
        pos_loss = tf.boolean_mask(conf_loss, pos_mask)
        neg_loss = tf.boolean_mask(conf_loss, neg_mask)

        # Top-k negative loss.
        pos_num = tf.reduce_sum(tf.cast(pos_mask, dtype=tf.int8))
        neg_num = tf.reduce_sum(tf.cast(neg_mask, dtype=tf.int8))
        neg_loss = tf.nn.top_k(neg_loss, tf.minimum(neg_num, 3*pos_num))[0]

        pos_loss = tf.reduce_sum(pos_loss)
        neg_loss = tf.reduce_sum(neg_loss)
        conf_loss = tf.negative(tf.add(pos_loss, neg_loss))

        return conf_loss

    def _locationLoss(self, pos_mask, anchors, gbbox, bboxes):
        """
        Compute location loss using ground truth bbox and predicted bbox.
        :param pos_mask: boolean positive mask.
        :param gbbox: ground truth bbox, shape[4]: cx, cy, width, height.
        :param bboxes: predicted bbox, shape[width, height, anchor_num, 4], 4: cx, cy, width, height.
        :return: Location loss.
        """

        for _ in range(3):
            gbbox = tf.expand_dims(gbbox, 0)

        c_offset = tf.divide(tf.subtract(gbbox[:, :, :, 0:2], anchors[:, :, :, 0:2]), anchors[:, :, :, 2:4])
        wh_offset = tf.log(tf.divide(gbbox[:, :, :, 2:4], anchors[:, :, :, 2:4]))
        gbbox = tf.concat([c_offset, wh_offset], axis=3)

        diff = tf.subtract(bboxes, gbbox)
        loc_loss = utils.smoothL1(diff)
        loc_loss = tf.boolean_mask(loc_loss, pos_mask)
        loc_loss = tf.reduce_sum(loc_loss)

        return loc_loss


    def loss(self, glabels, gbboxes, labels, bboxes):
        """
        Compute loss.
        :param glabels: ground truth labels.
        :param gbboxes: gound truth bounding boxes.
        :param labels: predicted labels.
        :param bboxes: predicted bounding boxes.
        :return: None
        """
        fm_conf = self._net_conf['featuremaps']
        anchors = self._anchors

        """
        Compute loss for each item in batch,
        sum them up to total loss.
        """
        for b in range(self._net_conf['batch_size']):
            glabel = glabels[b]
            gbbox = gbboxes[b]
            labels = labels[b]
            bboxes = bboxes[b]

            with tf.name_scope('LOSS'):
                # for each layer, compute loss
                for m, layer_conf in enumerate(fm_conf):
                    # compute jaccard overlap.
                    overlap = utils.jaccardIndex(tf.constant(anchors[layer_conf.layer_name], dtype=tf.float32),
                                                 gbbox)

                    # get positive and negtive mask accoding to overlap
                    pos_mask = utils.positiveMask(overlap)
                    neg_mask = tf.logical_not(pos_mask)

                    # count matched box number
                    pos_count = tf.reduce_sum(tf.cast(pos_mask, dtype=tf.int64))

                    # compute confidence loss and location loss
                    conf_loss = self._confLoss(pos_mask, neg_mask, labels)
                    loc_loss = self._locationLoss(pos_mask, anchors[layer_conf.layer_name], gbbox, bboxes)
                    loc_loss = tf.multiply(loc_loss, self._net_conf['alpha'])

                    cur_loss = tf.cond(tf.equal(pos_count, 0), lambda :tf.constant(0),
                                         tf.divide(tf.add(conf_loss, loc_loss), pos_count))

                    total_loss = tf.add(cur_loss)

        return total_loss