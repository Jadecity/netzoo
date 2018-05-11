import os
import tensorflow as tf
import json
import tensorlayer.visualize as vis
import numpy as np
import matplotlib.pyplot as plt

def _int64List_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.flatten()))

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

    def encodeAll(self, src_path, data_home, label_file, class_file, dest_path, cnt_max, preprocessor=None):
        """
        Read json files in src_path, read corresponding image file,
        and convert them to tfrecord format.
        :param src_path: directory which contains json files.
        :param data_home: home prefix to image name in json files.
        :param label_file: json file contains class and labels.
        :param class_file: json file contains image file name and its class name.
        :param dest_path: destination directory where to put tfrecords.
        :param cnt_max: max number of tfrecord in each file.
        :return: none.
        """
        with open(label_file, 'r') as label_file:
            label_dict = json.load(label_file)

        with open(class_file, 'r') as class_file:
            class_dict = json.load(class_file)

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
            img_size = np.array(img_size)

            labels = []
            bboxes = []
            for obj in ori_rcd['objects']:
                labels.append(label_dict[obj['label']])
                object = [obj['x1'], obj['y1'], obj['x2'], obj['y2']]
                bboxes.append(object)
            labels = np.array(labels)
            bboxes = np.array(bboxes)

            class_id = label_dict[class_dict[ori_rcd['imgname']]]

            # Preprocess image and bounding boxes.
            if None != preprocessor:
                img, img_size, bboxes = preprocessor(img, img_size, bboxes)

                # plt.imshow(img)
                # plt.draw()
                # plt.waitforbuttonpress()

            feature = {
                'image_name':_bytes_feature(tf.compat.as_bytes(ori_rcd['imgname'])),
                'image': _bytes_feature(img.tobytes()),
                'size': _int64List_feature(img_size),
                'class': _int64_feature(class_id),
                'labels': _int64List_feature(labels),
                'bbox_num': _int64_feature(np.shape(bboxes)[0]),
                'bboxes': _int64List_feature(bboxes)
            }

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Writing the serialized example.
            example_str = example.SerializeToString()
            writer.write(example_str)

            print('cnt: %d' % cnt)
            cnt = cnt + 1

            #write out records each cnt_max items
            if 0 == cnt % cnt_max:
                writer.close()
                tfrecord_filename = os.path.join( dest_path, '%d.tfrecords' % (cnt))
                writer = tf.python_io.TFRecordWriter(tfrecord_filename)

            # break
        writer.close()