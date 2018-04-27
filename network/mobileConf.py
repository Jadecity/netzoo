"""
Mobilenet configuration.
"""

def loadMobileConf():
    """
    Mobilenet configuration.
    :return: Mobilenet configuration.
    """
    conf = {}
    conf['width_mult'] = 1
    conf['input_size'] = 224
    conf['resolution_mult'] = conf['input_size'] / 224
    conf['is_train'] = False
    conf['class_num'] = 100

    return conf