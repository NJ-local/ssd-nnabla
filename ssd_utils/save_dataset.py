# -*- coding: utf-8 -*-
import numpy as np
import sys, os, glob

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

if __name__ == "__main__":
    # [load modules]
    sys.path.append('../')
    from ssd_models import *
    from ssd_utils import *
    from ssd_config import *
    from make_default_box import *
    from get_dataset import *
    # [hyper parameters]
    # [mac]
    g_imageset_path = '/Users/nj/work_station/machine_learning/dataset/pascal_VOC/VOCdevkit/VOC2007/JPEGImages/'
    g_annotationset_path = '/Users/nj/work_station/machine_learning/dataset/pascal_VOC/VOCdevkit/VOC2007/Annotations/'
    g_out_train_dataset_dir = '../../ssd_dataset/train/'
    g_out_test_dataset_dir = '../../ssd_dataset/test/'
    # # [Google Colab]
    # g_imageset_path = '../../JPEGImages/'
    # g_annotationset_path = '../../Annotations/'
    # g_out_train_dataset_dir = '../../ssd_dataset/train/'
    # g_out_test_dataset_dir = '../../ssd_dataset/test/'
    # [def model]
    train_x = nn.Variable(g_input_size)
    # (batch_size, box num * class ( or coordinate), default box y, default box x)
    train_fmap_list = ssd_network(train_x, g_box_num_list, g_class_num, test=False)
    train_fmap_conf_list = [fmap[0] for fmap in train_fmap_list]
    # [make default boxes]
    default_box_coordinate_list_list = make_default_boxes(train_fmap_conf_list, g_default_box_aspect_list_list)
    # default box
    for i, _default_box_coordinate_list in enumerate(default_box_coordinate_list_list):
        if i == 0:
            default_box_coordinate_list = _default_box_coordinate_list
        else:
            default_box_coordinate_list += _default_box_coordinate_list

    # [load dataset]
    my_tdata, my_vdata = get_dataset_from_source(g_imageset_path, g_annotationset_path)
    # train
    my_tdata_images_list = my_tdata['image']
    my_tdata_labels_list_source = my_tdata['label']
    print('convert training labels.')
    my_tdata_labels_list = trans_labelset_for_default_box(
        my_tdata_labels_list_source, 
        default_box_coordinate_list, 
        _class_num=g_class_num
    )
    # [save training data]
    my_tdata = []
    print('saving training dataset.')
    for i in range(len(my_tdata_images_list)):
        my_tdata.append(
            {
                'image': my_tdata_images_list[i], 
                'image_info': my_tdata_labels_list_source[i]['image_size_info'], 
                'label': my_tdata_labels_list[i]
            }
        )
    if not os.path.isdir(g_out_train_dataset_dir):
        os.makedirs(g_out_train_dataset_dir)
    save_dataset_npz(g_out_train_dataset_dir, my_tdata)
    # test
    my_vdata_images_list = my_vdata['image']
    my_vdata_labels_list_source = my_vdata['label']
    print('convert test labels.')
    my_vdata_labels_list = trans_labelset_for_default_box(
        my_vdata_labels_list_source, 
        default_box_coordinate_list, 
        _class_num=g_class_num
    )
    # [save test data]
    my_vdata = []
    print('saving test dataset.')
    for i in range(len(my_vdata_images_list)):
        my_vdata.append(
            {
                'image': my_vdata_images_list[i], 
                'image_info': my_vdata_labels_list_source[i]['image_size_info'], 
                'label': my_vdata_labels_list[i]
            }
        )
    if not os.path.isdir(g_out_test_dataset_dir):
        os.makedirs(g_out_test_dataset_dir)
    save_dataset_npz(g_out_test_dataset_dir, my_vdata)
