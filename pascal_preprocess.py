"""
Process annotations in pascal.
"""
import os.path as path
import xml.etree.ElementTree as ET
import glob
import json
import os

def annotation2json(ann_path, dest_json_path):
    """
    Convert all annotations in ann_path to standard json file in dest_json_path.
    :param ann_path: path to pascal annotations.
    :param dest_json_path: path to destination json files.
    :return: None.
    """
    class_list = {}

    # Check folder existance.
    if not path.isdir(ann_path):
        print('Path %s not exists!' % ann_path)

    if not path.isdir(dest_json_path):
        print('Path %s not exists!' % dest_json_path)

    for ann_full_name in glob.glob(path.join(ann_path, '*.xml')):
        ann_file = ann_full_name[ann_full_name.rfind('/') + 1:]
        dest_file = open(path.join(dest_json_path, ann_file.replace('xml', 'json')), mode='w')

        json_obj = {}
        tree = ET.parse(path.join(ann_path, ann_file))
        root = tree.getroot()
        json_obj['imgname'] = root.find('filename').text

        size = root.find('size')
        json_obj['imgsize'] = {'width':   int(size.find('width').text),
                               'height':  int(size.find('height').text),
                               'channel': int(size.find('depth').text)
                               }

        objects = root.findall('object')
        bboxes = []
        for obj in objects:
            bndbox = obj.find('bndbox')
            bboxes.append({
                'label':obj.find('name').text,
                'x1': int(bndbox.find('xmin').text),
                'y1': int(bndbox.find('ymin').text),
                'x2': int(bndbox.find('xmax').text),
                'y2': int(bndbox.find('ymax').text)
            })
            class_list[obj.find('name').text] = 0

        json_obj['objects'] = bboxes
        json.dump(json_obj, dest_file)
        dest_file.close()

    # write class labels to disk, just for once.
    keys = class_list.keys()

    for i, key in enumerate(keys):
        class_list[key] = i + 1

    class_label = open('/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/classes.json', 'w')
    json.dump(class_list, class_label)
    class_label.close()

def filename2label(label_file, txt_path, dest_path):
    """
    Parse relationship between file name and label name(for classification).
    :param label_file: filename of the label name json file.
    :param txt_path: directory containing train-val txt files.
    :param dest_path: directory to put relationship file.
    :return: None
    """

    label_file = open(label_file, 'r')
    label_dict = json.load(label_file)

    filename_label = {}
    for classname in label_dict.keys():
        filename = path.join(txt_path, classname + '_trainval.txt')
        for line in open(filename):
            imgfile, pos = line.split()
            if '1' == pos:
                filename_label[imgfile + '.jpg'] = classname

    out_json = path.join(dest_path, 'imgname_label.json')
    out_json = open(out_json, mode='w')
    json.dump(filename_label, out_json)
    out_json.close()



if __name__ == '__main__':
    ann_path = '/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/Annotations'
    dest_path = '/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/Annotations_json'

    # annotation2json(ann_path, dest_path)

    filename2label('/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/classes.json',
                   '/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/ImageSets/Main',
                   '/home/autel/libs/ssd-tensorflow-ljanyst/pascal-voc/trainval/VOCdevkit/VOC2007/')
    # a = ['a', 'b', 'c']
    # b = [x for x in a if x.endswith('a')]
    # print(b)
    # country_data_as_string = r"""<annotation>
    #                                 <filename>000138.jpg</filename>
    #                             </annotation>
    #                             """
    # root = ET.fromstring(country_data_as_string)
    # filename = root.find('filename')
    # print(filename.text)
    # a = {'a': 1, 'b':1}
    # for i, key in enumerate(a.keys()):
    #     print('i:%d, key:%s' % (i, key))