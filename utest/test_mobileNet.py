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


    path = '/home/autel/data/exp_imgs'
    img_name = 'face.jpg'
    img = tl.vis.read_image(img_name, path)
    img = img[:224, :224]
    img = np.reshape(img, [1, 224, 224, 3])
    img = tf.constant(np.ones([3, 224, 224, 3]), dtype=tf.float32)
    ft, _ = ft_extractor.predict(img)

    ss = tf.Session()

    # initialize all variables in the session
    tl.layers.initialize_global_variables(ss)

    # ft = ss.run(ft.outputs, feed_dict={input_img: img})
    ft = ss.run(ft.outputs)
    ss.close()
    print(np.shape(ft))
    # print(ft)

    # close session

