import network.mobileConf as mobileConf
import network.mobileNet as mobileNet
import network.ssdConf as ssdConf
import network.ssdNet as ssdNet
import tensorflow as tf
import tensorlayer as tl
import numpy as np

if __name__ == '__main__':
    # Create feature extractor.
    mobile_conf = mobileConf.loadMobileConf()
    ft_extractor = mobileNet.MobileNet(mobile_conf)

    ssd_conf = ssdConf.loadSSDConf()
    ssd_net = ssdNet.SSDNet(ssd_conf, ft_extractor)

    input_img = tf.placeholder(tf.float32, [1, 224, 224, 3])
    ft = ssd_net.predict(input_img)

    path = '/home/autel/data/exp_imgs'
    img_name = 'face.jpg'
    img = tl.vis.read_image(img_name, path)
    img = img[:224, :224]
    img = np.reshape(img, [1, 224, 224, 3])

    ss = tf.Session()

    # initialize all variables in the session
    tl.layers.initialize_global_variables(ss)

    ft = ss.run(ft.outputs, feed_dict={input_img: img})
    ss.close()

    print(ft)