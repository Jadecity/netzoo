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

    path = '/home/autel/data/exp_imgs'
    img_name = 'face.jpg'
    img = tl.vis.read_image(img_name, path)
    img = img[:224, :224]
    img = np.reshape(img, [1, 224, 224, 3])

    ss = tf.Session()

    # initialize all variables in the session
    tl.layers.initialize_global_variables(ss)

    pred, logits, locations = ss.run((pred, logits, locations), feed_dict={input_img: img})
    ss.close()

    print(np.shape(pred))