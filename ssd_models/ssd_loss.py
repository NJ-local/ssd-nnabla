# -*- coding: utf-8 -*-
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

def ssd_loss(_ssd_confs, _ssd_locs, _label, _alpha=1):
    # input
    # _ssd_confs : type=nn.Variable, prediction of class. shape=(batch_size, default boxes, class num + 1)
    # _ssd_locs : type=nn.Variable, prediction of location. shape=(batch_size, default boxes, 4)
    # _label : type=nn.Variable, shape=(batch_size, default boxes, class num + 1 + 4)
    # _alpha : type=float, hyperparameter. this is weight of loc_loss.

    # output
    # loss : type=nn.Variable

    def smooth_L1(__pred_locs, __label_locs):
        # input
        # __pred_locs : type=nn.Variable, 
        # __label_locs : type=nn.Variable, 

        # output
        # _loss : type=nn.Variable, loss of location.

        return F.mul_scalar(F.huber_loss(__pred_locs, __label_locs), 0.5)

    # _label_conf : type=nn.Variable, label of class. shape=(batch_size, default boxes, class num + 1) (after one_hot)
    # _label_loc : type=nn.Variable, label of location. shape=(batch_size, default boxes, 4)
    label_conf = F.slice(
        _label, 
        start=(0,0,4), 
        stop=_label.shape, 
        step=(1,1,1)
    )
    label_loc = F.slice(
        _label, 
        start=(0,0,0), 
        stop=(_label.shape[0], _label.shape[1], 4), 
        step=(1,1,1)
    )

    # conf
    ssd_pos_conf, ssd_neg_conf = ssd_separate_conf_pos_neg(_ssd_confs)
    label_conf_pos, _ = ssd_separate_conf_pos_neg(label_conf)
    # pos
    pos_loss = F.sum(
                        F.mul2(
                            F.softmax(ssd_pos_conf, axis=2), 
                            label_conf_pos
                        )
                        , axis=2
                    )
    # neg
    neg_loss = F.sum(F.log(ssd_neg_conf), axis=2)
    conf_loss = F.sum(F.sub2(pos_loss, neg_loss), axis=1)

    # loc
    pos_label = F.sum(label_conf_pos, axis=2)      # =1 (if there is sonething), =0 (if there is nothing)
    loc_loss = F.sum(F.mul2(F.sum(smooth_L1(_ssd_locs, label_loc), axis=2), pos_label), axis=1)

    # [2019/07/18]
    label_match_default_box_num = F.slice(
        _label, 
        start=(0,0,_label.shape[2] - 1), 
        stop=_label.shape, 
        step=(1,1,1)
    )
    label_match_default_box_num = F.sum(label_match_default_box_num, axis=1)
    label_match_default_box_num = F.r_sub_scalar(label_match_default_box_num, _label.shape[1])
    label_match_default_box_num = F.reshape(label_match_default_box_num, (label_match_default_box_num.shape[0],), inplace=False)
    # label_match_default_box_num : type=nn.Variable, inverse number of default boxes that matches with pos.

    # loss
    loss = F.mul2(F.add2(conf_loss, F.mul_scalar(loc_loss, _alpha)), label_match_default_box_num)
    loss = F.mean(loss)
    return loss

def ssd_separate_conf_pos_neg(_ssd_conf):
    # input
    # _ssd_conf : type=nn.Variable, shape=(batch_size, default boxes, pos num + neg num)

    # output
    # ssd_pos_conf : type=nn.Variable, shape=(batch_size, default boxes, pos num)
    # ssd_neg_conf : type=nn.Variable, shape=(batch_size, default boxes, neg num)

    ssd_pos_conf = F.slice(
                        _ssd_conf, 
                        start=(0,0,0),  
                        stop=(
                                _ssd_conf.shape[0], 
                                _ssd_conf.shape[1], 
                                _ssd_conf.shape[2] - 1
                            ), 
                        step=(1,1,1)
                        )
    ssd_neg_conf = F.slice(
                        _ssd_conf, 
                        start=(0,0,_ssd_conf.shape[2] - 1),  
                        stop=(
                                _ssd_conf.shape[0], 
                                _ssd_conf.shape[1], 
                                _ssd_conf.shape[2]
                            ), 
                        step=(1,1,1)
                        )
    return ssd_pos_conf, ssd_neg_conf

# F.less_scalarにbackward計算が実装されていなかったため、reluを用いて等価なものを実装する。
def my_less_scalar(_x, _scalar):
    # input
    # _x : type=nn.Variable
    # _scalar : type=float
    # output
    # flags : type=nn.Variable, same shape with _x

    temp = F.r_sub_scalar(_x, _scalar)
    temp = F.sign(temp, alpha=0)
    flags = F.relu(temp)
    return flags

