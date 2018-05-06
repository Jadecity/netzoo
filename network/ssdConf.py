"""
SSD parameter configuration.
"""
from collections import namedtuple

FeatmapConf = namedtuple('FeatmapConf', ['layer_name', 'size', 'scale', 'ratios'])

def addSSDConf(global_conf):
    conf = global_conf
    conf['featuremaps'] = [FeatmapConf(layer_name= 'layer-6',
                             size = 28,
                             scale = 0.1,
                             ratios= [1, 2, 1/2.0]),
                           FeatmapConf(layer_name='layer-12',
                             size = 14,
                             scale=0.2,
                             ratios=[1, 2, 3, 1 / 2.0, 1 / 3.0]),
                           FeatmapConf(layer_name='block-1',
                             size = 7,
                             scale=0.375,
                             ratios=[1, 2, 3, 1 / 2.0, 1 / 3.0]),
                           FeatmapConf(layer_name='block-2',
                             size = 4,
                             scale=0.55,
                             ratios=[1, 2, 3, 1 / 2.0, 1 / 3.0]),
                           FeatmapConf(layer_name='block-3',
                             size = 2,
                             scale=0.725,
                             ratios=[1, 2, 1 / 2.0]),
                           FeatmapConf(layer_name='block-4',
                             size = 1,
                             scale=0.9,
                             ratios=[1, 2, 1 / 2.0])
                           ]

    conf['alpha'] = 0.8
    return conf

