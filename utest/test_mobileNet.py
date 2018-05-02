import network.mobileConf as mobileConf
import network.mobileNet as mobileNet
import common.config as gconf
import tensorflow as tf
import tensorlayer as tl
import numpy as np

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Create feature extractor.
    global_conf = gconf.loadTrainConf()
    ft_extractor = mobileNet.MobileNet(global_conf)

    input_img = tf.placeholder(tf.float32, [1, 224, 224, 3])
    ft, shape = ft_extractor.predict(input_img)

    path = '/home/autel/data/exp_imgs'
    img_name = 'face.jpg'
    img = tl.vis.read_image(img_name, path)
    img = img[:224, :224]
    img = np.reshape(img, [1, 224, 224, 3])

    ss = tf.Session()

    # initialize all variables in the session
    tl.layers.initialize_global_variables(ss)

    # ft = ss.run(ft.outputs, feed_dict={input_img: img})
    shape = ss.run(shape, feed_dict={input_img: img})
    ss.close()
    print(np.shape(shape))
    # print(ft)

    # close session

