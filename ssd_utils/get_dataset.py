# -*- coding: utf-8 -*-
import cv2
import pickle
import numpy as np
import glob, sys, os

import xml.etree.ElementTree as ET      # get_annotation_from_xml_one()

g_seed = 0
g_label_index_dict = {
    'aeroplane': 0, 
    'bicycle': 1, 
    'bird': 2, 
    'boat': 3, 
    'bottle': 4, 
    'bus': 5, 
    'car': 6, 
    'cat': 7, 
    'chair': 8, 
    'cow': 9, 
    'diningtable': 10, 
    'dog': 11, 
    'horse': 12, 
    'motorbike': 13, 
    'person': 14, 
    'pottedplant': 15, 
    'sheep': 16, 
    'sofa': 17, 
    'train': 18, 
    'tvmonitor': 19
}

def get_data_batch(_dataset_dict, _start_index, _batch_size, _data_num):
    # input
    # _dataset_dict : type=dict of list, key='image', 'label', 'image_info', value=list, this is the output of load_dataset_npz().
    # _start_index : type=int
    # _batch_size : type=int
    # _data_num : type=int, =len(_dataset_dict['**'])

    # output
    # out_data_list : type=dict of data, key='image', 'label', 'image_info', value=images, labels, image_infos
    # next_start_index : type=int, next start index

    if _start_index >= _data_num:
        print('----- error -----')
        print('error code: _start_index >= _data_num')
        print('_start_index : {}'.format(_start_index))
        print('_data_num : {}'.format(_data_num))
        exit()
    elif _start_index + _batch_size < _data_num:
        next_start_index = _start_index + _batch_size
        images = _dataset_dict['image'][_start_index : next_start_index]
        labels = _dataset_dict['label'][_start_index : next_start_index]
        image_infos = _dataset_dict['image_info'][_start_index : next_start_index]
    else:
        next_start_index = _data_num - _start_index
        images = _dataset_dict['image'][_start_index : _data_num] + _dataset_dict['image'][0 : next_start_index]
        labels = _dataset_dict['label'][_start_index : _data_num] + _dataset_dict['label'][0 : next_start_index]
        image_infos = _dataset_dict['image_info'][_start_index : _data_num] + _dataset_dict['image_info'][0 : next_start_index]
    out_data_list = {
        'image': images, 
        'label': labels, 
        'image_info': image_infos
    }
    return out_data_list, next_start_index

def save_dataset_npz(_outfiledir, _dataset_dict_list):
    # input
    # _outfiledir : type=string
    # _dataset_dict : type=list of dict, key='image' or 'label', value=np.ndarray or dict(key='image_size_info' or 'label_list', value=dict or list)

    # process
    # save dataset as _filename

    save_data_num = len(_dataset_dict_list)
    for i, dataset_dict in enumerate(_dataset_dict_list):
        np.savez(_outfiledir + '{:06d}.npz'.format(i), **dataset_dict)
        sys.stdout.write('\rnow saving ... {}/{}'.format(i + 1, save_data_num))
        sys.stdout.flush()
    print('')

def load_dataset_npz(_dataset_dir):
    # input
    # _dataset_dir : type=string, 

    # output
    # out_dataset : type=dict of list, key='image' or 'label' or 'image_info', value=list of np.ndarray

    out_data_image_list = []
    out_data_label_list = []
    out_data_image_info_list = []

    data_path_list = glob.glob(_dataset_dir + '*')
    data_path_num = len(data_path_list)
    for i, data_path in enumerate(data_path_list):
        # load
        data = np.load(data_path)       # data.keys() = ['image', 'image_info', 'label']
        data.allow_pickle = True
        # set
        out_data_image_list.append(data['image'])
        out_data_label_list.append(data['label'])
        out_data_image_info_list.append(data['image_info'])
        sys.stdout.write('\rnow loading ... {}/{}'.format(i + 1, data_path_num))
        sys.stdout.flush()
    print('')

    out_dataset = {
        'image': out_data_image_list, 
        'label': out_data_label_list, 
        'image_info': out_data_image_info_list
    }

    return out_dataset

def get_label_one(_image_size_info, _annotation_list, _label_index_dict=g_label_index_dict):
    # input
    # _image_size_info : type=dict
    # _annotation_list : type=list of dict
    # _label_index_dict : type=dict of label index, key=label name, value=index

    # output
    # label_list : type=list of list, each list length = 4 + 20

    height = _image_size_info['height']
    width = _image_size_info['width']

    label_list = []
    for annotation in _annotation_list:
        label_name = annotation['name']
        if label_name not in _label_index_dict.keys():
            print('----- error -----')
            print('{} does not exist in bellow.'.format(label_name))
            for key in _label_index_dict.keys():
                print(key)
            print('-----------------')
            exit()
        else:
            # index
            index = _label_index_dict[label_name]
            index_list = [0 for i in range(len(_label_index_dict))]
            index_list[index] = 1
            # source
            ymin = annotation['point1'][0]
            xmin = annotation['point1'][1]
            ymax = annotation['point2'][0]
            xmax = annotation['point2'][1]
            # label
            y_center = (ymin + ymax) / 2.0
            x_center = (xmin + xmax) / 2.0
            label_height = ymax - y_center
            label_width = xmax - x_center
            coordinate_list = [
                y_center / height, 
                x_center / width, 
                label_height / height, 
                label_width / width
            ]
            # concatenate
            label = coordinate_list + index_list
            label_list.append(np.array(label))

    return label_list

def get_annotation_from_xml_one(_annotation_path):
    # input
    # _annotation_path : type=string, path of annotation (image name).xml

    # output
    # image_size_info : type=dict
    # annotation_list : type=list of dict

    tree = ET.parse(_annotation_path)
    objects = tree.findall('object')

    image_size_info = {
        'height': int(tree.findall('size/height')[0].text), 
        'width': int(tree.findall('size/width')[0].text), 
        'channels': int(tree.findall('size/depth')[0].text)
    }

    annotation_list = []
    for name in objects:
        annotation_list.append(
            {
                'name': name.findall('name')[0].text, 
                'point1': (
                    int(name.findall('bndbox/ymin')[0].text), 
                    int(name.findall('bndbox/xmin')[0].text)
                    ), 
                'point2': (
                    int(name.findall('bndbox/ymax')[0].text), 
                    int(name.findall('bndbox/xmax')[0].text)
                    )
            }
        )
    return image_size_info, annotation_list

# [2019/06/30]  データセットdirを読み込んで、画像と、annotationを取得するように変更する。
def get_dataset_from_source(_imageset_path, _annotationset_path, _train_dataset_rate=0.8, _image_shape=(300,300)):
    # input
    # _imageset_path : type=string, there are image files(.jpg) under this path.
    # _annotationset_path : type=string, there are annotation files(.xml) under this path.
    # _train_dataset_rate : type=float, rate of train dataset fo all.
    # _image_shape : type=list of int, input image size of DNN.

    # output
    # dataset : type=list of dict, key='image' or 'label', value=image or label.

    def append_data(_image_list, _label_list, _image_path, _annotation_path, __image_shape):
        # input
        # _image_list : type=list of image
        # _label_list : type=list of label
        # _image_path : type=string, 
        # _annotation_path : type=string
        # __image_shape : type=list of int, image shape for resize

        # process
        # append image to _image_list and label to _label_list

        # image
        image = cv2.imread(_image_path)
        image = cv2.resize(image, __image_shape)
        image = np.transpose(image, [2,0,1])
        image = image.astype('float32')
        image /= 255.0
        _image_list.append(image)
        # label
        image_size_info, annotation_list = get_annotation_from_xml_one(_annotation_path)
        label_list = get_label_one(image_size_info, annotation_list)
        _label_list.append(
            {'image_size_info': image_size_info, 
            'label_list': label_list}
        )

    # seed for pick up train data.
    np.random.seed(g_seed)

    # data list
    image_path_list = glob.glob(_imageset_path + '*.jpg')
    data_num = len(image_path_list)

    # make dataset
    train_data_index_list = np.random.choice([i for i in range(data_num)], int(data_num * _train_dataset_rate), replace=False)
    # train
    train_image_list = []
    train_label_list = []
    # test
    test_image_list = []
    test_label_list = []
    for index in range(len(image_path_list)):
        image_path = image_path_list[index]
        image_name = image_path.replace(_imageset_path, '')
        annotation_name = image_name.replace('.jpg', '.xml')
        annotation_path = _annotationset_path + annotation_name
        if index in train_data_index_list:
            # train data
            append_data(
                train_image_list, 
                train_label_list, 
                image_path, 
                annotation_path, 
                _image_shape
            )
        else:
            # test data
            append_data(
                test_image_list, 
                test_label_list, 
                image_path, 
                annotation_path, 
                _image_shape
            )

    dataset = [
        {
            'image': train_image_list, 
            'label': train_label_list
        }, 
        {
            'image': test_image_list, 
            'label': test_label_list
        }
    ]

    return dataset
