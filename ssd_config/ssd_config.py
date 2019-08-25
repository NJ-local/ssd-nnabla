# -*- coding: utf-8 -*-

# [google colab]
g_train_data_path = '../ssd_dataset/train/'
g_test_data_path = '../ssd_dataset/test/'
# g_basenet_source_path = './SSD/SSD_nnabla_model/feature_extractor_VGG_params.h5'
g_context = 'cudnn'
g_device_id = 0
# [mac]
# g_imageset_path = '/Users/nj/work_station/machine_learning/dataset/pascal_VOC/VOCdevkit/VOC2007/JPEGImages/'
# g_annotationset_path = '/Users/nj/work_station/machine_learning/dataset/pascal_VOC/VOCdevkit/VOC2007/Annotations/'
# [2019/07/17]
# g_train_data_path = '../ssd_dataset/train/'
# g_test_data_path = '../ssd_dataset/test/'
# g_basenet_source_path = './SSD_nnabla_model/feature_extractor_VGG_params.h5'
# g_context = 'cpu'

g_batch_size = 50
g_input_size = (g_batch_size, 3, 300, 300)
g_default_box_aspect_list_list = [
    [1.0, 2.0, 1.0/2.0],                    # fmap1
    [1.0, 2.0, 3.0, 1.0/2.0, 1.0/3.0],      # fmap2
    [1.0, 2.0, 3.0, 1.0/2.0, 1.0/3.0],      # fmap3
    [1.0, 2.0, 3.0, 1.0/2.0, 1.0/3.0],      # fmap4
    [1.0, 2.0, 3.0, 1.0/2.0, 1.0/3.0],      # fmap5
    [1.0, 2.0, 3.0, 1.0/2.0, 1.0/3.0]       # fmap6
]
g_box_num_list = [len(boxes) + 1 for boxes in g_default_box_aspect_list_list]
# g_box_num_list = [4,6,6,6,6,6]
g_out_dir = './result/'
g_out_default_boxes_dir = g_out_dir + 'default_boxes/'
g_class_num = 21
g_ssd_loss_alpha = 1.0
g_valid_interval_epoch = 1

# [optimizer]
g_optimizer = 'SGD'
g_default_learning_rate = 0.5
g_weight_decay = 0.0001

# [monitor]
g_save_log_dir = g_out_dir + 'log/'
g_save_model_dir = g_out_dir + 'models/'
g_monitor_interval = 10

# [training]
g_max_epoch = 200

