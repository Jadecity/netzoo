"""
SSD parameter configuration.
"""
from collections import namedtuple

FeatmapConf = namedtuple('FeatmapConf', ['layer_name', 'size', 'scale', 'ratios'])

def addSSDConf(global_conf):
    conf = global_conf
    conf['featuremaps'] = {'layer-6':FeatmapConf(layer_name= 'layer-6',
                                       size = '28',
                                       scale = '',
                                       ratios= [1, 2, 1/2.0]),
                           # 'layer-12':FeatmapConf(layer_name='layer-12',
                           #             size='14',
                           #             scale='',
                           #             ratios=[1, 2, 3, 1 / 2.0, 1 / 3.0]),
                           # 'block-1':FeatmapConf(layer_name='block-1',
                           #             size='7',
                           #             scale='',
                           #             ratios=[1, 2, 3, 1 / 2.0, 1 / 3.0]),
                           # 'block-2':FeatmapConf(layer_name='block-2',
                           #             size='4',
                           #             scale='',
                           #             ratios=[1, 2, 3, 1 / 2.0, 1 / 3.0]),
                           # 'block-3':FeatmapConf(layer_name='block-3',
                           #             size='2',
                           #             scale='',
                           #             ratios=[1, 2, 1 / 2.0]),
                           # 'block-4':FeatmapConf(layer_name='block-4',
                           #             size='1',
                           #             scale='',
                           #             ratios=[1, 2, 1 / 2.0])
                           }
    return conf

