import network.mobileNet as mobileNet
import network.ssdConf as ssdConf
import network.ssdNet as ssdNet
import common.config as confutil
import tensorflow as tf
import tensorlayer as tl
import numpy as np

g_conf = confutil.loadTrainConf()

# Create feature extractor.
ft_extractor = mobileNet.MobileNet(g_conf)
ssd_conf = ssdConf.addSSDConf(g_conf)

ssd_net = ssdNet.SSDNet(ssd_conf, ft_extractor)
anchors = ssd_net._create_anchors(ssd_net._net_conf['featuremaps'])
