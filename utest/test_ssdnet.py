import network.mobileNet as mobileNet
import network.ssdConf as ssdConf
import network.ssdNet as ssdNet
import common.config as confutil
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import common.utils as utils

if __name__ == '__main__':
    g_conf = confutil.loadTrainConf()

    # Create feature extractor.
    ft_extractor = mobileNet.MobileNet(g_conf)

    ssd_conf = ssdConf.addSSDConf(g_conf)
    ssd_net = ssdNet.SSDNet(ssd_conf, ft_extractor)

    input_img = tf.placeholder(tf.float32, [1, 224, 224, 3])
    pred, logits, locations, endpoints = ssd_net.predict(input_img)

    # Compute loss and put them to tf.collections.LOSS and other loss.
    input_labels = tf.placeholder(tf.float32, [None, ssd_conf['class_num']])
    input_bboxes = tf.placeholder(tf.float32, [None, 4])
    total_loss = ssd_net.loss(pred_labels=logits, pred_bboxes=locations, glabels=input_labels, gbboxes=input_bboxes)

    path = '/home/autel/data/exp_imgs'
    img_name = 'face.jpg'
    img = tl.vis.read_image(img_name, path)
    img = img[:224, :224]
    img = np.reshape(img, [1, 224, 224, 3])

    # glabels = np.zeros([10])
    # glabels[4] = 1
    # glabels = tf.constant(glabels)
    # glabels = tf.reshape(glabels, shape=[1, 10])
    #
    # gboxes = np.array([0.2, 0.2, 0.1, 0.2])
    # gboxes = tf.constant(gboxes, dtype=tf.float32)
    # gboxes = tf.reshape(gboxes, shape=[1, 4])

    # Create optimizer.
    optimizer = utils.getOptimizer(ssd_conf)

    train_op = optimizer.minimize(total_loss)

    # initialize all variables in the session
    init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()

    with tf.Session() as ss:
        ss.run([init, local_init])
        for itr_idx in range(1):
            for i in range(len(ssd_conf['featuremaps'])):
                ss.run(train_op, feed_dict={input_img: img,
                                                input_labels: [[0,1,0,0,0]],
                                                input_bboxes:[[0.2, 0.2, 0.1, 0.2]]})

                # (pos_num, neg_num, neg_loss) = ss.run((ssd_net._val['pos_num'][i],
                #         ssd_net._val['neg_num'][i],
                #         ssd_net._val['neg_loss'][i]), feed_dict={input_img: img,
                #                                 input_labels: [[0,1,0,0,0]],
                #                                 input_bboxes:[[0.2, 0.2, 0.1, 0.2]]})
                #
                # print('%d--------' % i)
                # print(pos_num)
                # print(neg_num)
                # print(neg_loss)

    # gbboxes = [0.2, 0.2, 0.1, 0.2]
    # utils.visualizeAnchors(ssd_net._anchors, ssd_conf, gbboxes)

    # utils.visualizeOverlap(ssd_net._anchors, ssd_conf, gbboxes)
