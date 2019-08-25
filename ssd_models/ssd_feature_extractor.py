# -*- coding: utf-8 -*-
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

def vgg16(x):
    # Input:x -> 3,300,300

    # VGG11/MulScalar
    h = F.mul_scalar(x, 0.01735)
    # VGG11/AddScalar
    h = F.add_scalar(h, -1.99)
    # VGG11/Convolution -> 64,300,300
    h = PF.convolution(h, 64, (3,3), (1,1), name='Convolution')
    # VGG11/ReLU
    h = F.relu(h, True)
    # VGG11/MaxPooling -> 64,150,150
    h = F.max_pooling(h, (2,2), (2,2))
    # VGG11/Convolution_3 -> 128,150,150
    h = PF.convolution(h, 128, (3,3), (1,1), name='Convolution_3')
    # VGG11/ReLU_3
    h = F.relu(h, True)
    # VGG11/MaxPooling_2 -> 128,75,75
    h = F.max_pooling(h, (2,2), (2,2))
    # VGG11/Convolution_5 -> 256,75,75
    h = PF.convolution(h, 256, (3,3), (1,1), name='Convolution_5')
    # VGG11/ReLU_5
    h = F.relu(h, True)
    # VGG11/Convolution_6
    h = PF.convolution(h, 256, (3,3), (1,1), name='Convolution_6')
    # VGG11/ReLU_6
    h = F.relu(h, True)
    # VGG11/MaxPooling_3 -> 256,38,38
    h = F.max_pooling(h, (2,2), (2,2), True, (1,1))
    # VGG11/Convolution_8 -> 512,38,38
    h = PF.convolution(h, 512, (3,3), (1,1), name='Convolution_8')
    # VGG11/ReLU_8
    h = F.relu(h, True)
    # VGG11/Convolution_9
    h = PF.convolution(h, 512, (3,3), (1,1), name='Convolution_9')
    # VGG11/ReLU_9
    h = F.relu(h, True)
    # # VGG11/MaxPooling_4 -> 512,19,19
    # h = F.max_pooling(h, (2,2), (2,2))
    # # VGG11/Convolution_11
    # h = PF.convolution(h, 512, (3,3), (1,1), name='Convolution_11')
    # # VGG11/ReLU_11
    # h = F.relu(h, True)
    # # VGG11/Convolution_12
    # h = PF.convolution(h, 512, (3,3), (1,1), name='Convolution_12')
    # # VGG11/ReLU_12
    # h = F.relu(h, True)
    return h

