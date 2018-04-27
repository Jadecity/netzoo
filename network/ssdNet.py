"""
Reimplementation of SSD[1]
[1]: SSD: Single Shot MultiBox Detector
"""
import tensorlayer as tl
import tensorflow as tf

class SSDNet:
    def __init__(self, net_conf, feature_extractor):
        """
        Build a SSD network using a backbone feature extractor.
        :param net_conf: SSD network configuration.
        :param feature_extractor: backbone feature extractor.
        :return None
        """

        self._ft_extractor = feature_extractor
        self._net_conf = net_conf
        self._buildNet()

    def predict(self, input_img, is_training=False):
        """
        Predict labels and bboxes in image.
        :param input_img: input image.
        :param is_training: True during training period, otherwise false.
        :return: labels , corresponding bboxes, confidential and endpoints of the network.
        """
        self._input = input_img

        if is_training:
            # TODO remove dropout
            pass

        labels = self._labels
        bboxes = self._bboxes
        confidence = self._confidence
        endpoints = self._endpoints

        return labels, bboxes, confidence, endpoints

    def _buildNet(self):
        # add auxiliary conv layers to the end of backbone network

        # add classifier conv layers to each added feature map(including the last layer of backbone network)



        pass

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