#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import time
import os
import scipy
import tensorflow as tf
import multiprocessing
import keras
import argparse

from keras.models    import Sequential
from keras.models    import Model, load_model
from keras.layers    import Dense, Dropout, Flatten, BatchNormalization, Input, Embedding, LSTM, GRU
from keras.layers    import Conv2D, MaxPooling2D, AveragePooling2D, Bidirectional, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras           import backend as K
from keras.utils     import to_categorical

from sklearn.utils           import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics         import log_loss
from sklearn.preprocessing   import OneHotEncoder

#from sklearn.preprocessing import normalize

#tf.enable_eager_execution()


# In[5]:


test_sets = np.load('./test_sets.pkl')
train_test_index = 4


# In[6]:


# test model on current test set
test_set = test_sets[train_test_index]

# load model to allow for input of np array from TFRecords

# NOTE* this reloading of the model is redundant, but the author found 
# no way to consume TFRecords in the testing of keras model 
smodel = tf.keras.models.load_model('best_best_rescnnqstft3_b128.h5')


# In[7]:


# Thanks to https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/18_TFRecords_Dataset_API.ipynb

# Data in the form of TFRecords needs to be parsed before it is consumed by model
# In this case, the mp3 spectrograms and one hot encoded labels have been
# serialized into binary strings

# parse reads the serialized data

def parse(serialized):
    # Define a dict with the data-names and types we expect to find in the
    # TFRecords file. It is a bit awkward that this needs to be specified again,
    # because it could have been written in the header of the TFRecords file
    # instead.
    features =         {
            'image': tf.FixedLenSequenceFeature([], tf.float32, allow_missing = True),
            'label': tf.FixedLenSequenceFeature([], tf.float32, allow_missing = True)
        }
    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example (
                                                    serialized = serialized,
                                                    features   = features
                                                )
    image = parsed_example['image']

    # Get the label associated with the image.
    label = parsed_example['label']

    # The image and label are now correct TensorFlow types.
    return image, label


# In[8]:


# will create a tf.dataset for optimized consumption of training data
# the returned image, label tuple represents a call to iterator.get_next()

def create_dataset(filepath):
    
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)
    
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(parse, num_parallel_calls=8)
    
    # This dataset will go on forever
    dataset = dataset.repeat(1)
    
    # Set the number of datapoints you want to load and shuffle 
    dataset = dataset.shuffle(100)
    
    # Set the batchsize
    dataset = dataset.batch(1)
    
    # Create an iterator
    iterator = dataset.make_one_shot_iterator()
    
    # Create your tf representation of the iterator
    next_element = iterator.get_next()

#     # Bring your picture back in shape
#     image = tf.reshape(image, [-1, 128, 128, 1])
    
#     # Create a one hot array for your labels
#     #label = tf.one_hot(label, 5)
#     label = tf.reshape(label, [-1,5])

    
    return next_element


# In[9]:


len(test_sets[train_test_index])


# In[10]:


# load TFRecord data from test set and convert to np.array to consumed by 
# model for testing
acc = []
next_element = create_dataset(test_sets[train_test_index])

for i in range(37):
    print(i)

    with tf.Session() as sess:
    
        test_element = sess.run(next_element)
        image = np.asarray(test_element[0]).reshape(150,128,128,1)
        label = np.asarray(test_element[1]).reshape(150,5)
    res = smodel.test_on_batch(image, label)
    acc.append(res[1])


# In[11]:


expt_folder_name = 'pool/'
results_name     = ('pool_' + 
                    str('MaxPooling2D').replace('.','') +
                    'fold_' + 
                    str(train_test_index+1) + 
                    '_test_acc' + 
                    '.pkl')
results_path     = './results/' + expt_folder_name + results_name

print(results_path)


# In[12]:


np.mean(acc)


# In[13]:


np.asarray(acc).dump(results_path)


# In[ ]:


















































































# In[ ]:




