# -*- coding:utf-8 -*-
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

import numpy as np

from ssd_models.ssd_extra_feature_network import ssd_extra_feature_network
from ssd_models.ssd_feature_extractor import vgg16
# from my_utils.get_variables import GET_VARIABLES

def ssd_network(x, _box_num_list=[4,6,6,6,6,6], _class_num=21, test=False):
    # input
    # x : type=nn.Variable, shape=(batch_size, 3, 300, 300)
    # _box_num_list : type=list of int, default boxes number list.
    # _class_num : type=int, class number of objects.

    # output
    # type=list of nn.Variable, fmap.shape=(batch_size, box num * classes & coordinates, default box y, default box x)

    feature = vgg16(x)
    feature_list = ssd_extra_feature_network(feature, test)

    # extra feature map 1
    fmap1 = [F.relu(PF.convolution(
                    feature_list[0], 
                    outmaps=_box_num_list[0]*(_class_num), 
                    kernel=(3,3), 
                    pad=(1,1), 
                    stride=(1,1), 
                    name='fmap1_conf'
                    )), 
            F.relu(PF.convolution(
                    feature_list[0], 
                    outmaps=_box_num_list[0]*(4), 
                    kernel=(3,3), 
                    pad=(1,1), 
                    stride=(1,1), 
                    name='fmap1_loc'
                    ))]
    # extra feature map 2
    fmap2 = [F.relu(PF.convolution(
                    feature_list[1], 
                    outmaps=_box_num_list[1]*(_class_num), 
                    kernel=(3,3), 
                    pad=(1,1), 
                    stride=(1,1), 
                    name='fmap2_conf'
                    )), 
            F.relu(PF.convolution(
                    feature_list[1], 
                    outmaps=_box_num_list[1]*(4), 
                    kernel=(3,3), 
                    pad=(1,1), 
                    stride=(1,1), 
                    name='fmap2_loc'
                    ))]
    # extra feature map 3
    fmap3 = [F.relu(PF.convolution(
                    feature_list[2], 
                    outmaps=_box_num_list[2]*(_class_num), 
                    kernel=(3,3), 
                    pad=(1,1), 
                    stride=(1,1), 
                    name='fmap3_conf'
                    )), 
            F.relu(PF.convolution(
                    feature_list[2], 
                    outmaps=_box_num_list[2]*(4), 
                    kernel=(3,3), 
                    pad=(1,1), 
                    stride=(1,1), 
                    name='fmap3_loc'
                    ))]
    # extra feature map 4
    fmap4 = [F.relu(PF.convolution(
                    feature_list[3], 
                    outmaps=_box_num_list[3]*(_class_num), 
                    kernel=(3,3), 
                    pad=(1,1), 
                    stride=(1,1), 
                    name='fmap4_conf'
                    )), 
            F.relu(PF.convolution(
                    feature_list[3], 
                    outmaps=_box_num_list[3]*(4), 
                    kernel=(3,3), 
                    pad=(1,1), 
                    stride=(1,1), 
                    name='fmap4_loc'
                    ))]
    # extra feature map 5
    fmap5 = [F.relu(PF.convolution(
                    feature_list[4], 
                    outmaps=_box_num_list[4]*(_class_num), 
                    kernel=(3,3), 
                    pad=(1,1), 
                    stride=(1,1), 
                    name='fmap5_conf'
                    )), 
            F.relu(PF.convolution(
                    feature_list[4], 
                    outmaps=_box_num_list[4]*(4), 
                    kernel=(3,3), 
                    pad=(1,1), 
                    stride=(1,1), 
                    name='fmap5_loc'
                    ))]
    # extra feature map 6
    fmap6 = [F.relu(PF.convolution(
                    feature_list[5], 
                    outmaps=_box_num_list[5]*(_class_num), 
                    kernel=(1,1), 
                    stride=(1,1), 
                    name='fmap6_conf'
                    )), 
            F.relu(PF.convolution(
                    feature_list[5], 
                    outmaps=_box_num_list[5]*(4), 
                    kernel=(1,1), 
                    stride=(1,1), 
                    name='fmap6_loc'
                    ))]
    
    return fmap1, fmap2, fmap3, fmap4, fmap5, fmap6
    # re_fmap_list = ssd_fmap_reshape([fmap1, fmap2, fmap3, fmap4, fmap5, fmap6])

def ssd_fmap_reshape(_fmap_list):
    # input
    # _fmap_list : type=list of nn.Variable, fmap.shape=(batch_size, box num * classes & coordinates, default box y, default box x)

    # output
    # _re_fmap_list : type=list of nn.Variable, shape=(batch_size, box num * classes & coordinates, default boxes))

    re_fmap_conf_list = []
    re_fmap_loc_list = []
    index = 0
    for fmap in _fmap_list:
        # shape=(batch_size, class_num or coordinates, default box y, default box x)
        # ---> =(batch_size, class_num or coordinates, default box y * default box x)
        re_fmap_conf_list.append(F.reshape(fmap[0], (fmap[0].shape[0], fmap[0].shape[1], np.prod(fmap[0].shape[2:]))))
        re_fmap_loc_list.append(F.reshape(fmap[1], (fmap[1].shape[0], fmap[1].shape[1], np.prod(fmap[1].shape[2:]))))
        # shape=(batch_size, class_num or coordinates, default box y * default box x)
        # ---> =(batch_size, default box y * default box x, class_num or coordinates)
        re_fmap_conf_list[index] = F.transpose(re_fmap_conf_list[index], (0,2,1))
        re_fmap_loc_list[index] = F.transpose(re_fmap_loc_list[index], (0,2,1))
        index += 1

    return re_fmap_conf_list, re_fmap_loc_list

def ssd_fmap_concat(_re_fmap_list, _map_num):
    # input
    # _re_fmap_list : type=list of nn.Variable, shape=(batch_size, box num * classes & coordinates, default boxes))
    # _map_num : type=int, class num or coordinate num(=4).

    # output
    # ssd_output : type=nn.Variable, shape=(batch_size, box num * default boxes * classes( or coordinates), _map_num)

    for i in range(len(_re_fmap_list)):
        if i == 0:
            ssd_output = F.reshape(
                                _re_fmap_list[i], 
                                (
                                    _re_fmap_list[i].shape[0], 
                                    np.prod(_re_fmap_list[i].shape[1:])
                                    )
                                )
        else:
            ssd_output = F.concatenate(
                                ssd_output, 
                                F.reshape(
                                _re_fmap_list[i], 
                                (
                                    _re_fmap_list[i].shape[0], 
                                    np.prod(_re_fmap_list[i].shape[1:])
                                    )
                                ), 
                                axis=1
                                )
    ssd_output = F.reshape(
                        ssd_output, 
                        (
                            ssd_output.shape[0], 
                            int(ssd_output.shape[1] / _map_num), 
                            _map_num
                        )
    )
    return ssd_output

