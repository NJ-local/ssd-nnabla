# -*- coding: utf-8 -*-
import numpy as np
import cv2

def make_default_boxes(_fmap_list, _aspect_list_list, _image_height=300, _s_min=0.2, _s_max=0.9):
    # input
    # _fmap_list : type=list of nn.Variable, feature maps
    # _aspect_list_list : type=list of aspect list of features. (width:height = width/height)
    # _s_min : type=float
    # _s_max : type=float
    # _image_height : type=int, height of input image

    # output
    # default_box_coordinate_list : type=list of default boxes (default box = (y, x, height, width))

    def calc_scale(k):
        return _s_min + ((_s_max - _s_min) / (m - 1)) * (k - 1)

    default_box_coordinate_list = []
    m = len(_fmap_list)
    for i, fmap in enumerate(_fmap_list):
        default_box_coordinate_list.append([])
        # index of scale
        k = i + 1
        # scale
        scale_k = calc_scale(k)
        # calc width and height of default box for fmap index i
        # real width = width * _image_height
        # real height = height * _image_height
        for j, aspect in enumerate(_aspect_list_list[i]):
            if aspect == 1:
                width_list = [scale_k, np.sqrt(scale_k*calc_scale(k + 1))]
                height_list = [scale_k, np.sqrt(scale_k*calc_scale(k + 1))]
            elif aspect > 1:
                width_list = [scale_k]
                height_list = [scale_k / aspect]
            elif aspect < 1:
                width_list = [scale_k * aspect]
                height_list = [scale_k]
            else:
                print('----- error -----')
                print('k = {}'.format(k))
                print('scale_k = {}'.format(scale_k))
                print('aspect = {}'.format(aspect))
                print('-----------------¥n')
            # calc coordinate ratios for fmap index i
            for y in range(fmap.shape[2]):
                for x in range(fmap.shape[3]):
                    for ii, width in enumerate(width_list):
                        height = height_list[ii]
                        Y = (y + y + 1) / (2 * fmap.shape[2])
                        X = (x + x + 1) / (2 * fmap.shape[3])
                        # default_box_coordinate_list[i].append([Y, X, height, width])
                        if X <= 1 and Y <= 1:
                            default_box_coordinate_list[i].append([Y, X, height, width])
                        else:
                            print('----- error -----')
                            print('k = {}'.format(k))
                            print('scale_k = {}'.format(scale_k))
                            print('aspect = {}'.format(aspect))
                            print('Y = {}'.format(Y))
                            print('X = {}'.format(X))
                            print('-----------------¥n')
    return default_box_coordinate_list

def draw_default_box_one(_image, _default_box, _colar=(0,0,255)):
    # input
    # _image : type=np.ndarray, shape=(height, width, channels)
    # _default_box : type=list of float, value=[y ratio, x ratio, height ratio, width ratio]
    # _colar : type=tuple of int, colar of rectangle.

    # process
    # draw _default_box to _image

    image_height = _image.shape[0]
    image_width = _image.shape[1]

    default_box_y = int(image_height * _default_box[0])
    default_box_x = int(image_width * _default_box[1])
    default_box_height = int(image_height * _default_box[2] / 2)
    default_box_width = int(image_width * _default_box[3] / 2)

    point1 = (default_box_y - default_box_height, default_box_x - default_box_width)
    point2 = (default_box_y + default_box_height, default_box_x + default_box_width)

    cv2.rectangle(_image, point1, point2, _colar, 1)

def draw_default_boxes(_image, _default_boxes_list):
    # input
    # _image : type=np.ndarray, shape=(height, width, channels)
    # _default_boxes_list : type=list of list of float, value=[y ratio, x ratio, height ratio, width ratio]

    # process
    # draw _default_box to _image

    scale = 100
    for i, default_box in enumerate(_default_boxes_list):
        colar_1 = int(scale*i/(255 * 255))
        colar_2 = int((scale * i - colar_1 * 255 * 255) / 255)
        colar_3 = int(scale * i - 255*255*colar_1 - 255*colar_2)
        colar = (colar_1, colar_2, colar_3)
        draw_default_box_one(_image, default_box, colar)



