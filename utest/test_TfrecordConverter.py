import common.tfrecordConverter as converter
import json

if __name__ == '__main__':
    src_path = '/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/Annotations_json'
    data_home = '/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/JPEGImages'
    dest_path = '/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/tfrecords'
    label = '/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/classes.json'

    cvtr = converter.TfrecordConverter()
    cvtr.encodeAll(src_path, data_home, label, dest_path, 1000)

    # json.loads(r'{"imgname":"Train/pos/person_240.png", \
    # "imgsize": {"width" :"640","height": "480","channel": "3"},\
    # "objects":{"label":"PASperson","x1":"319","y1":"78","x2":"444","y2":"446"}\
    # }')

    # json.loads(r'{"imgname": "Train/pos/person_and_bike_209.png","imgsize": 	  {"width" :640,	   "height": 480,	   "channel": 3	  },"objects":[{"label": "PASperson","x1": 336,"y1": 112,"x2": 418,"y2": 316}]}')

