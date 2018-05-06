import network.mobileNet as mobileNet
import network.ssdConf as ssdConf
import network.ssdNet as ssdNet
import common.config as confutil
import tensorflow as tf
import tensorlayer as tl
import numpy as np

if __name__ == '__main__':
    g_conf = confutil.loadTrainConf()

    # Create feature extractor.
    ft_extractor = mobileNet.MobileNet(g_conf)

    ssd_conf = ssdConf.addSSDConf(g_conf)
    ssd_net = ssdNet.SSDNet(ssd_conf, ft_extractor)

    input_img = tf.placeholder(tf.float32, [1, 224, 224, 3])
    pred, logits, locations, endpoints = ssd_net.predict(input_img)

    # Compute loss and put them to tf.collections.LOSS and other loss.
    input_labels = tf.placeholder(tf.int8, [None, ssd_conf['class_num']])
    input_bboxes = tf.placeholder(tf.int8, [None, 4])
    total_loss = ssd_net.loss(labels=logits, bboxes=locations, glabels=input_labels, gbboxes=input_bboxes)

    path = '/home/autel/data/exp_imgs'
    img_name = 'face.jpg'
    img = tl.vis.read_image(img_name, path)
    img = img[:224, :224]
    img = np.reshape(img, [1, 224, 224, 3])

    glabels = np.zeros([10])
    glabels[4] = 1
    glabels = tf.constant(glabels)
    glabels = tf.reshape(glabels, shape=[1, 10])

    gboxes = np.array([0.2, 0.2, 0.1, 0.2])
    gboxes = tf.constant(gboxes)
    gboxes = tf.reshape(gboxes, shape=[1, 4])

    ss = tf.Session()

    # initialize all variables in the session
    tl.layers.initialize_global_variables(ss)

    total_loss = ss.run((total_loss), feed_dict={input_img: img,
                                                 input_labels: [[0,1]],
                                                 input_bboxes:[[0.2, 0.2, 0.1, 0.2]]})
    ss.close()

    print(total_loss)
