# -*- coding: utf-8 -*-
import numpy as np
import sys

def trans_labelset_for_default_box(_labelset, _default_box_coordinate_list, _class_num=21, _jaccard_threshould=0.5):
    # input
    # _labelset : type=list of dict, the list index is same to image index. dict key='image_size_info' or 'label_list', value=dict or list
    # _default_box_coordinate_list : type=list of all default boxes

    # output
    # labelset : type=list of label, label shape=[len(default boxes number)][4(coordinate) + class number + 1]

    labelset = []
    trans_data_num = len(_labelset)
    for i, label_dict in enumerate(_labelset):
        labelset.append(
            get_label_for_default_box_one(
                label_dict['label_list'], 
                _default_box_coordinate_list, 
                _class_num=_class_num, 
                _jaccard_threshould=_jaccard_threshould
            )
        )
        sys.stdout.write('\rnow converting ... {}/{}'.format(i + 1, trans_data_num))
        sys.stdout.flush()
    print('')
    return labelset

def get_label_for_default_box_one(_label_from_annotation_list, _default_box_coordinate_list, _class_num=21, _jaccard_threshould=0.5):
    # input
    # _label_from_annotation_list : type=list of np.ndarray, each shape=(24,), (y_center, x_center, height, width, classes(20,))
    # _default_box_coordinate_list : type=list of list, = (Y, X, height, width)

    # output
    # label_for_default_box : type=list of list, same of _default_box_list_list

    label_for_default_box = []
    for default_box_coordinate in _default_box_coordinate_list:
        label_init = [0 for i in range(_class_num)]
        max_jaccard_coef = -1
        max_jaccard_index = -1
        for i, label_coordinate in enumerate(_label_from_annotation_list):
            jaccard_coef = get_jaccard_one(label_coordinate, default_box_coordinate)
            if max_jaccard_coef < jaccard_coef:
                max_jaccard_coef = jaccard_coef
                max_jaccard_index = i
        if max_jaccard_coef <= 0.5:
            # neg label
            label_init[_class_num - 1] = 1
            label_for_default_box.append([0,0,0,0] + label_init)
        else:
            # pos label
            label_coordinate = _label_from_annotation_list[max_jaccard_index]
            delta_y_center = (label_coordinate[0] - default_box_coordinate[0]) / default_box_coordinate[2]
            delta_x_center = (label_coordinate[1] - default_box_coordinate[1]) / default_box_coordinate[3]
            delta_height = np.log(label_coordinate[1] / default_box_coordinate[1])
            delta_width = np.log(label_coordinate[2] / default_box_coordinate[2])
            label_for_default_box.append(
                [
                    delta_y_center, 
                    delta_x_center, 
                    delta_height, 
                    delta_width
                ] + list(label_coordinate[4:]) + [0]
            )
    return label_for_default_box

def get_jaccard_one(_label_coordinate, _default_box_coordinate):
    # input
    # _label_coordinate : type=list, = (Y, X, height, width)
    # _default_box_coordinate : type=list, = (Y, X, height, width)

    # output
    # jaccard : type=float

    label_point_min = (_label_coordinate[0] - _label_coordinate[2] / 2.0, _label_coordinate[1] - _label_coordinate[3] / 2.0)
    label_point_max = (_label_coordinate[0] + _label_coordinate[2] / 2.0, _label_coordinate[1] + _label_coordinate[3] / 2.0)
    default_box_point_min = (_default_box_coordinate[0] - _default_box_coordinate[2] / 2.0, _default_box_coordinate[1] - _default_box_coordinate[3] / 2.0)
    default_box_point_max = (_default_box_coordinate[0] + _default_box_coordinate[2] / 2.0, _default_box_coordinate[1] + _default_box_coordinate[3] / 2.0)
    
    if default_box_point_max[1] <= label_point_min[1] or label_point_max[1] <= default_box_point_min[1]:
        jaccard = 0.0
    elif default_box_point_max[0] <= label_point_min[0] or label_point_max[0] <= default_box_point_min[0]:
        jaccard = 0.0
    else:
        # X
        if label_point_min[1] <= default_box_point_min[1]:
            point_A = [label_point_min, label_point_max]
            point_B = [default_box_point_min, default_box_point_max]
        else:
            point_A = [default_box_point_min, default_box_point_max]
            point_B = [label_point_min, label_point_max]
        # Y
        if point_A[0][0] <= point_B[0][0]:
            temp = point_A
            point_A = point_B[0]        # upper left of B
            point_B = temp[1]           # lower right of A
        else:
            temp = point_A
            point_A = (point_B[1][0], point_B[0][1])
            point_B = (temp[0][0], temp[1][1])
        # calc
        AandB = np.abs((point_A[0] - point_B[0]) * (point_A[1] - point_B[1]))
        default_box_square = np.abs(
            (default_box_point_max[0] - default_box_point_min[0]) * (default_box_point_max[1] - default_box_point_min[1])
        )
        label_square = np.abs(
            (label_point_max[0] - label_point_min[0]) * (label_point_max[1] - label_point_min[1])
        )
        AorB = default_box_square + label_square - AandB
        jaccard = float(AandB) / float(AorB)
    return jaccard

# def set_label_for_var(_label_batch_list, _default_box_coordinate_list, _class_label, _loc_label):
#     # input
#     # _label_batch_list : type=list of nd.array, list of class and loc labels
#     # _default_box_coordinate_list : type=list of list, = (Y, X, height, width)
#     # _class_label : type=nn.Variable, shape=(batch_size, default boxes num, class num)
#     # _loc_label : type=nn.Variable, shape=(batch_size, default boxes num, 4)

#     # process
#     # set label for _class_label and _loc_label from _label_batch_list



