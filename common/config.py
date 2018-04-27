"""
Define all configurations for training, evaluation and test.
"""

def loadTrainConf():
    """
    Define all configuration params for training period.
    :return: configuration params.
    """
    train_conf = {}
    train_conf['batch_size'] = 50
    train_conf['epoch_num'] = 20

    train_conf['optimizer'] = 'AdamOptimizer'
    train_conf['learning_rate'] = 0.001

    train_conf['dataset_path'] = '/home/autel/data/INRIAPerson/Train/tfrecords/pos'

    return train_conf