from __future__ import division

import six
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,EarlyStopping,ReduceLROnPlateau,TensorBoard
from keras import backend as keras
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from scipy.stats import kendalltau  
import os
os.chdir("D:/htic/breastpathq/datasets/")
import numpy as np
import glob
import cv2
from tqdm import tqdm
import csv
import csv
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import *
import cv2, numpy as np
def result_model(input_size=(1,),weights=None):
    model=Sequential()
    model.add(Dense(15,activation="relu",input_shape=input_size))
    model.add(Dense(7,activation="relu"))
    #model.add(Flatten())
    model.add(Dense(1,activation="sigmoid"))
    model.compile(optimizer= Adam(lr=3e-3),loss='mean_squared_error')

    if(weights):
        model.load_weights(weights)
    return model
with open('train_labels.csv', mode='r') as infile:
    reader = csv.reader(infile)
    with open('train_new.csv', mode='w') as outfile:
        writer = csv.writer(outfile)
        mydict = {str(rows[0])+"_"+str(rows[1]):rows[2] for rows in reader}
del mydict['slide_rid']
import csv
with open('val_labels.csv', mode='r') as infile:
    reader = csv.reader(infile)
    with open('val_new.csv', mode='w') as outfile:
        writer = csv.writer(outfile)
        valdict = {str(rows[0])+"_"+str(rows[1]):rows[2] for rows in reader}
del valdict['slide_rid']        


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier



class ResnetBuilder(object):
    
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions,pretrained_weights=None):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        HUBER_DELTA = 0.5
        def smoothL1(y_true, y_pred):
            x   = K.abs(y_true - y_pred)
            x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
            return  K.sum(x)
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[0], input_shape[1], input_shape[2])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="sigmoid")(flatten1)

        model = Model(inputs=input, outputs=dense)
        model.compile(optimizer = Adam(lr = 3e-4), loss = 'mean_squared_error')
        if(pretrained_weights):
            model.load_weights(pretrained_weights)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs,weights=None):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2],pretrained_weights=weights)

    @staticmethod
    def build_resnet_34(input_shape, num_outputs,weights=None):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3],pretrained_weights=weights)

    @staticmethod
    def build_resnet_50(input_shape, num_outputs,weights=None):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3],pretrained_weights=weights)

    @staticmethod
    def build_resnet_101(input_shape, num_outputs,weights=None):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3],pretrained_weights=weights)

    @staticmethod
    def build_resnet_152(input_shape, num_outputs,weights=None):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3],pretrained_weights=weights)
import glob


model1=ResnetBuilder. build_resnet_152(input_shape=(256,256,3),num_outputs=1)

model1.load_weights("resnet152/fulltrain.29-0.03-0.0205.hdf5")
imgdir=glob.glob("D:/htic/breastpathq/breastpathq-test/test1/*.tif")
result=np.zeros((1119,1))
n=0
for img in tqdm(imgdir):
    imgname=os.path.basename(img)
    iname=os.path.splitext(imgname)[0]
    imag=cv2.imread(img)
    imag = np.reshape(imag,(1,)+imag.shape)
    result[n]=model1.predict(imag, batch_size=None, verbose=0, steps=None)
    n+=1
res=[]    
model=result_model()
model.load_weights("result_net/result.07-0.02-0.0168.hdf5")
for i in result:
    i = np.reshape(i,(1,)+i.shape)   
    res1= model.predict(i,batch_size=None, verbose=0, steps=None)
    res.append(res1)
res=np.array(res)

n=0
'''
csvData = [['slide', 'rid','y','p']]
for img in tqdm(imgdir):
    imgname=os.path.basename(img)
    iname=os.path.splitext(imgname)[0]
    imag=cv2.imread(img)
    imag = np.reshape(imag,(1,)+imag.shape)
    #result[n]=model.predict(imag, batch_size=None, verbose=0, steps=None)
    csvData.append([iname[0:6],iname[7:],valdict[iname],res[n][0][0]])
    n+=1
with open('test/val_Results.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)

csvFile.close()                        
import csv
'''
'''
with open('val_Results.csv', mode='r') as infile:
    reader = csv.reader(infile)
    with open('val_err.csv', mode='w') as outfile:
        writer = csv.writer(outfile)
        hist = {str(rows[0])+"_"+str(rows[1]):rows[4] for rows in reader}
del hist['slide_rid']
err=np.zeros((185,1))
n=0
names=[]
for i in hist:
    err[n]=hist[i]
    names.append(hist[i])
    n+=1
from matplotlib import pyplot as plt
plt.hist(err,bins=20)
'''
csvData = [['slide', 'rid','p']]
n=0
for img in tqdm(imgdir):
    imgname=os.path.basename(img)
    iname=os.path.splitext(imgname)[0]
    imag=cv2.imread(img)
    imag = np.reshape(imag,(1,)+imag.shape)
    
    csvData.append([iname[0:6],iname[7:],res[n][0][0]])
    n+=1
import csv


with open('test/Winterfell_Results.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)

csvFile.close()
