# -*- coding: utf-8 -*-
import nnabla.functions as F
import numpy as np

from .get_dataset import *
from .ssd_utils import *

# def get_values(_conf, _loc, _label, _jaccard_threshould=0.5):
# 	# input
# 	# _conf : type=np.ndarray, shape=(class num + 1,)
# 	# _loc : type=np.ndarray, shape=(4,)
# 	# _label : type=np.ndarray, shape=(class num + 4 + 1,)

# 	# output
# 	# TP : type=int or None, true positive label. for precision
# 	# FP : type=int or None, false positive label. for precision
# 	# FN : type=int or None, false positive label. for recall

# 	# [2019/07/23]	precision location の調整が必要
# 	jaccard = get_jaccard_one(_label[:4], _loc)
# 	if jaccard < _jaccard_threshould:
# 		TP = None
# 		FP = np.argmax(_conf)		# if FP and FN = class num + 1, this is true precision.
# 		FN = np.argmax(_loc[4:])	# if FN and FN = class num + 1, this is true precision.
# 	else:
# 		TP = 


def get_AP(_ssd_input, _ssd_confs, _ssd_locs, _ssd_label, _vdata, _jaccard_threshould=0.5):
    # input
    # _ssd_input : type=nn.Variable, shape=(batch_size, 3, 300, 300)
    # _ssd_confs : type=nn.Variable, prediction of class. shape=(batch_size, default boxes, class num + 1)
    # _ssd_locs : type=nn.Variable, prediction of location. shape=(batch_size, default boxes, 4)
    # _ssd_label : type=nn.Variable, shape=(batch_size, 1)
    # _vdata : type=list, validation dataset

    # output
    # AP : type=float

    ssd_output = F.concatenate(_ssd_locs, _ssd_confs, axis=1)

    vdata_num = len(_vdata['image'])
    # val_iter = int(np.ceil(vdata_num / _ssd_confs.shape[0]))
    val_iter = int(np.floor(vdata_num / _ssd_confs.shape[0]))

    # pred_class = np.argmax(_ssd_confs.d, axis=2)

    start_index = 0
    for i in range(val_iter):
        data_batch, start_index = get_data_batch(_vdata, start_index, _ssd_confs.shape[0], vdata_num)
        image_batch = np.array(data_batch['image'])      # shape=(batch_size, 3, 300, 300)
        label_batch = np.array(data_batch['label'])

        # [set & forward]
        _ssd_input.d = image_batch
        _ssd_label.d = label_batch
        ssd_output.forward(clear_buffer=True)


