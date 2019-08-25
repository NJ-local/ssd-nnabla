# -*- coding:utf-8 -*-
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

def ssd_extra_feature_network(x, test=False):
    # ex. (batch_size, 512, 38, 38) -> (batch_size, 1024, 19, 19)
    conv6 = PF.convolution(x, outmaps=1024, kernel=(3,3), pad=(1,1), stride=(1,1), name='conv6')
    conv6 = F.relu(conv6)
    conv6 = PF.batch_normalization(conv6, batch_stat=not test, name='conv6_bn')
    conv7 = F.max_pooling(conv6, kernel=(2,2), stride=(2,2))
    conv7 = PF.convolution(conv7, outmaps=1024, kernel=(1,1), stride=(1,1), name='conv7')
    conv7 = F.relu(conv7)
    conv7 = PF.batch_normalization(conv7, batch_stat=not test, name='conv7_bn')
    # ex. (batch_size, 1024, 19, 19) -> (batch_size, 512, 10, 10)
    conv8 = PF.convolution(conv7, outmaps=256, kernel=(1,1), name='conv8_1')
    conv8 = F.relu(conv8)
    conv8 = PF.batch_normalization(conv8, batch_stat=not test, name='conv8_1_bn')
    conv8 = PF.convolution(conv8, outmaps=512, kernel=(3,3), pad=(1,1), stride=(2,2), name='conv8_2')
    conv8 = F.relu(conv8)
    conv8 = PF.batch_normalization(conv8, batch_stat=not test, name='conv8_2_bn')
    # ex. (batch_size, 512, 10, 10) -> (batch_size, 256, 5, 5)
    conv9 = PF.convolution(conv8, outmaps=128, kernel=(1,1), name='conv9_1')
    conv9 = F.relu(conv9)
    conv9 = PF.batch_normalization(conv9, batch_stat=not test, name='conv9_1_bn')
    conv9 = PF.convolution(conv9, outmaps=256, kernel=(3,3), pad=(1,1), stride=(2,2), name='conv9_2')
    conv9 = F.relu(conv9)
    conv9 = PF.batch_normalization(conv9, batch_stat=not test, name='conv9_2_bn')
    # ex. (batch_size, 256, 5, 5) -> (batch_size, 256, 3, 3)
    conv10 = PF.convolution(conv9, outmaps=128, kernel=(1,1), name='conv10_1')
    conv10 = F.relu(conv10)
    conv10 = PF.batch_normalization(conv10, batch_stat=not test, name='conv10_1_bn')
    conv10 = PF.convolution(conv10, outmaps=256, kernel=(3,3), pad=(1,1), stride=(2,2), name='conv10_2')
    conv10 = F.relu(conv10)
    conv10 = PF.batch_normalization(conv10, batch_stat=not test, name='conv10_2_bn')
    # ex. (batch_size, 256, 3, 3) -> (batch_size, 256, 1, 1)
    conv11 = PF.convolution(conv10, outmaps=128, kernel=(1,1), name='conv11_1')
    conv11 = F.relu(conv11)
    conv11 = PF.batch_normalization(conv11, batch_stat=not test, name='conv11_1_bn')
    conv11 = PF.convolution(conv11, outmaps=256, kernel=(3,3), stride=(3,3), name='conv11_2')
    conv11 = F.relu(conv11)
    conv11 = PF.batch_normalization(conv11, batch_stat=not test, name='conv11_2_bn')

    return [conv6, conv7, conv8, conv9, conv10, conv11]

