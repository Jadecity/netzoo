import tensorflow as tf


def getOptimizer(opt_conf):
    """
    Create an optimizer using config.
    :param opt_conf: optimizer configuration
    :return: tf.train.Optimizer object or None
    """
    optimizer = None

    if opt_conf['opt_name'] == 'AdamOptimizer':
        optimizer = tf.train.AdamOptimizer(learning_rate=opt_conf['learning_rate'],
                                           epsilon=opt_conf['epsilon'])

    return optimizer

