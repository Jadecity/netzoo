import common.tfrecordConverter as converter
import json

if __name__ == '__main__':
    src_path = '/home/autel/data/INRIAPerson/Train/annotations_json'
    data_home = '/home/autel/data/INRIAPerson/'
    dest_path = '/home/autel/data/INRIAPerson/Train/tfrecords/pos'

    cvtr = converter.TfrecordConverter()
    cvtr.encodeAll(src_path, data_home, dest_path, 1000)

    # json.loads(r'{"imgname":"Train/pos/person_240.png", \
    # "imgsize": {"width" :"640","height": "480","channel": "3"},\
    # "objects":{"label":"PASperson","x1":"319","y1":"78","x2":"444","y2":"446"}\
    # }')

    # json.loads(r'{"imgname": "Train/pos/person_and_bike_209.png","imgsize": 	  {"width" :640,	   "height": 480,	   "channel": 3	  },"objects":[{"label": "PASperson","x1": 336,"y1": 112,"x2": 418,"y2": 316}]}')

