"""
Define all configurations for training, evaluation and test.
"""

def loadTrainConf():
    """
    Define all configuration params for training period.
    :return: configuration params.
    """
    train_conf = {}
    train_conf['class_num'] = 5
    train_conf['batch_size'] = 1
    train_conf['epoch_num'] = 20

    train_conf['optimizer'] = 'AdamOptimizer'
    train_conf['epsilon'] = 1e-8
    train_conf['learning_rate'] = 0.001

    train_conf['dataset_path'] = '/home/autel/data/INRIAPerson/Train/tfrecords/pos'

    train_conf['input_size'] = 224
    train_conf['resolution_mult'] = train_conf['input_size'] / 224
    train_conf['is_train'] = False

    # conf for mobile net
    train_conf['width_mult'] = 1

    return train_conf