#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import time
#import fma
import os
#import librosa
import scipy
import tensorflow as tf
#from tqdm import tqdm
import multiprocessing
from sklearn.utils import shuffle
from sklearn.metrics import log_loss

from sklearn.preprocessing import OneHotEncoder

import keras
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input, Embedding, LSTM, GRU
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Bidirectional, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K
from keras.utils import to_categorical

from sklearn.utils import shuffle
#from sklearn.preprocessing import normalize

import argparse

#tf.enable_eager_execution()


# In[2]:


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# In[23]:


model = load_model('./best_psy_cnn.h5')


# In[24]:


layer_dict = dict([(layer.name, layer) for layer in model.layers])


# In[25]:


layer_dict


# In[ ]:





# In[26]:


# thanks to https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py

# dimensions of the generated pictures for each filter.
img_width = 128
img_height = 128

# name the convolutional layer to be visualized  
layer_name = 'conv2d_3'
input_img = model.input


# In[27]:


# the following produces visualizations of the most activated filters in a convolutional
# layer
kept_filters = []
for filter_index in range(200):
    # we only scan through the first 200 filters,
    # but there are actually 512 of them
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 1.

    # we start from a gray image with some random noise
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 1, img_width, img_height))
    else:
        input_img_data = np.random.random((1, img_width, img_height, 1))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # we run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))


# In[53]:


len(kept_filters)


# In[56]:


plt.imshow(kept_filters[3][1].reshape(128,128))


# In[59]:


from keras.preprocessing.image import save_img
import matplotlib.pyplot as plt

# we will stich the best 64 filters on a 8 x 8 grid.
n = 2

# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top 64 filters.
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

# build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 5
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

# fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        width_margin = (img_width + margin) * i
        height_margin = (img_height + margin) * j
        stitched_filters[
            width_margin: width_margin + img_width,
            height_margin: height_margin + img_height, :] = img
        
        

# save the result to disk
save_img('stitched_filters_%dx%d_%s.png' % (n, n, layer_name), stitched_filters)


# In[ ]:




