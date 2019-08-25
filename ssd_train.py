# -*- coding: utf-8 -*-
import nnabla as nn
import nnabla.solvers as S
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed

import numpy as np
import cv2
import os

from ssd_models import *
from ssd_utils import *
from ssd_config import *

if __name__ == "__main__":
    # [calc core extension]
    if g_context == 'cudnn':
        from nnabla.ext_utils import get_extension_context
        ctx = get_extension_context(g_context, device_id=g_device_id, type_config='float')
        nn.set_default_context(ctx)

    # [make output dir]
    if not os.path.isdir(g_out_dir):
        os.makedirs(g_out_dir)

    # [def model]
    train_x = nn.Variable(g_input_size)
    # (batch_size, box num * class ( or coordinate), default box y, default box x)
    train_fmap_list = ssd_network(train_x, g_box_num_list, g_class_num, test=False)
    train_fmap_conf_list = [fmap[0] for fmap in train_fmap_list]
    # (batch_size, default boxes, box num * class ( or coordinate))
    re_fmap_conf_list, re_fmap_loc_list = ssd_fmap_reshape(ssd_network(train_x, g_box_num_list, g_class_num, test=False))
    # (batch_size, all default boxes * (box num * class ( or coordinate)))
    ssd_conf_output = ssd_fmap_concat(re_fmap_conf_list, g_class_num)
    ssd_loc_output = ssd_fmap_concat(re_fmap_loc_list, 4)
    # coef
    coef_dict_source = nn.get_parameters(grad_only=False)
    # ----- print -----
    print('ssd_conf_output = {}'.format(ssd_conf_output))
    print('ssd_loc_output = {}'.format(ssd_loc_output))
    total_box_num = 0
    for i, fmap in enumerate(re_fmap_conf_list):
        print('[{}] : {}'.format(i, fmap.shape))
        total_box_num += fmap.shape[1] * g_box_num_list[i]
    print('total_box_num = {}'.format(total_box_num))

    # [make default boxes]
    default_box_coordinate_list_list = make_default_boxes(train_fmap_conf_list, g_default_box_aspect_list_list)
    # ----- print -----
    total_box_num = 0
    for i, default_box_coordinate in enumerate(default_box_coordinate_list_list):
        total_box_num += len(default_box_coordinate)
        print('default_box[{}] : {}'.format(i, len(default_box_coordinate)))
    print('total_box_num = {}'.format(total_box_num))
    # ----- draw default boxes -----
    if not os.path.isdir(g_out_default_boxes_dir):
        os.makedirs(g_out_default_boxes_dir)
    for i, default_box_coordinate in enumerate(default_box_coordinate_list_list):
        image = np.ones((300,300,3)).astype(np.uint8)
        image.fill(255)
        draw_default_boxes(image, default_box_coordinate)
        out_default_box_path = g_out_default_boxes_dir + 'default_box_{}.png'.format(i)
        cv2.imwrite(out_default_box_path, image)
    # default box
    for i, _default_box_coordinate_list in enumerate(default_box_coordinate_list_list):
        if i == 0:
            default_box_coordinate_list = _default_box_coordinate_list
        else:
            default_box_coordinate_list += _default_box_coordinate_list


    # [def loss]
    label = nn.Variable([g_batch_size, total_box_num, g_class_num + 4])
    ssd_train_loss = ssd_loss(
                            ssd_conf_output, 
                            ssd_loc_output, 
                            label, 
                            _alpha=g_ssd_loss_alpha
                            )
    print('ssd_train_loss = {}'.format(ssd_train_loss))

    # [get dataset]
    my_tdata = load_dataset_npz(g_train_data_path)
    my_vdata = load_dataset_npz(g_test_data_path)
    tdata_num = len(my_tdata['image'])
    iter_num_max = int(np.ceil(tdata_num / g_batch_size))

    # [def solver]
    if g_optimizer == 'SGD':
        solver = S.Sgd(g_default_learning_rate)
    elif g_optimizer == 'Adam':
        solver = S.Adam(g_default_learning_rate)
    elif g_optimizer == 'AdaBound':
        solver = S.AdaBound(g_default_learning_rate)
    solver.set_parameters(coef_dict_source)

    # [def monitor]
    monitor = Monitor(g_save_log_dir)
    monitor_loss = MonitorSeries("Training loss", monitor, interval=g_monitor_interval)
    monitor_time = MonitorTimeElapsed("Training time", monitor, interval=g_monitor_interval)
    monitor_verr = MonitorSeries("Validation error", monitor, interval=1)

    # [training]
    iter_num = 0
    start_index = 0
    newest_model = 'dummy'
    for epoch in range(g_max_epoch):
        for i in range(iter_num_max):
            data_batch, start_index = get_data_batch(my_tdata, start_index, g_batch_size, tdata_num)
            image_batch = np.array(data_batch['image'])      # shape=(batch_size, 3, 300, 300)
            label_batch = np.array(data_batch['label'])

            # [set data]
            train_x.d = image_batch
            label.d = label_batch

            # [forward/backward]
            ssd_train_loss.forward()
            solver.zero_grad()
            ssd_train_loss.backward()

            # [solver update]
            solver.weight_decay(g_weight_decay)
            solver.update()

            # [monitor]
            monitor_loss.add(iter_num, ssd_train_loss.d.copy())
            monitor_time.add(iter_num)
            
            # [count up]
            iter_num += 1
        
        # [validation]
        if os.path.isfile(newest_model):
            os.remove(newest_model)
        newest_model = g_save_model_dir + 'newest_epoch_{}_iter{}.h5'.format(epoch, iter_num)
        

