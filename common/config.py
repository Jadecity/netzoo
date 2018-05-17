"""
Define all configurations for training, evaluation and test.
"""

def loadTrainConf():
    """
    Define all configuration params for training period.
    :return: configuration params.
    """
    train_conf = {}
    train_conf['class_num'] = 10
    train_conf['batch_size'] = 32
    train_conf['epoch_num'] = 300

    train_conf['optimizer'] = 'AdamOptimizer'
    train_conf['epsilon'] = 1e-8
    train_conf['learning_rate'] = 0.1
    train_conf['weight_decay'] = 0.0005
    train_conf['momentum'] = 0.9

    # train_conf['dataset_path'] = '/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/tfrecords'
    train_conf[
        'dataset_path'] = '/home/autel/data/cifar/cifar-10-batches-py/tfrecords'
    train_conf['mean_img'] = '/home/autel/data/cifar/cifar-10-batches-py/mean_img.npy'

    train_conf['input_size'] = 32
    train_conf['resolution_mult'] = train_conf['input_size'] / 224
    train_conf['is_train'] = True

    # conf for mobile net
    train_conf['width_mult'] = 1

    # conf for resNeXt29
    train_conf['card'] = 16

    # Tensor board log config
    train_conf['log_step'] = 10 # 10 batch log once
    train_conf['log_dir'] = '/home/autel/PycharmProjects/netzoo/log'


    return train_conf