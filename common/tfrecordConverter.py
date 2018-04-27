import os
import tensorflow as tf
import json
import tensorlayer.visualize as vis
import numpy as np


def _int64List_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[x for x in value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# _bytes is used for string/char values
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class TfrecordConverter():
    """
    Tools to convert json files and images to tfrecord format.
    """
    def __init__(self):
        pass

    def encodeAll(self, src_path, data_home, dest_path, cnt_max):
        """
        Read json files in src_path, read corresponding image file,
        and convert them to tfrecord format.
        :param src_path: directory which contains json files.
        :param data_home: home prefix to image name in json files.
        :param dest_path: destination directory where to put tfrecords.
        :param cnt_max: max number of tfrecord in each file.
        :return: none.
        """
        label_dict = {'PASperson': 1}

        cnt = 1

        tfrecord_filename = os.path.join(dest_path, '%d.tfrecords' % (cnt))
        writer = tf.python_io.TFRecordWriter(tfrecord_filename)

        #process each json file
        for file in os.listdir(src_path):
            content = open(os.path.join(src_path, file)).read()
            ori_rcd = json.loads(content)

            img = vis.read_image(ori_rcd['imgname'], data_home)
            img_size = [ori_rcd['imgsize']['width'],
                        ori_rcd['imgsize']['height'],
                        ori_rcd['imgsize']['channel']]

            for obj in ori_rcd['objects']:
                labels = label_dict[obj['label']]
                bboxes = [obj['x1'], obj['y1'], obj['x2'], obj['y2']]

                feature = {
                    'image': _bytes_feature(img.tostring()),
                    'size': _int64List_feature(img_size),
                    'labels': _int64_feature(labels),
                    'bboxes': _int64List_feature(bboxes)
                }

                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Writing the serialized example.
                writer.write(example.SerializeToString())

                cnt = cnt + 1

                #write out records each cnt_max items
                if 0 == cnt % cnt_max:
                    writer.close()
                    tfrecord_filename = os.path.join( dest_path, '%d.tfrecords' % (cnt))
                    writer = tf.python_io.TFRecordWriter(tfrecord_filename)

        writer.close()