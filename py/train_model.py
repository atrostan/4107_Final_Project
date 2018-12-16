#!/usr/bin/env python
# coding: utf-8

# In[10]:


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


# In[11]:


train_sets = np.load('./train_sets.pkl')


# In[12]:


# certain index of train-test split

train_test_index = 4


# In[13]:


len(train_sets[train_test_index])


# In[14]:


K.tensorflow_backend._get_available_gpus()


# In[15]:


path = '../Final/tf_records/'

img_shape    = (128,128)
img_size     = 128
num_channels = 1
num_classes  = 5

# retrieve all the TFRecord file paths

from os import listdir
from os.path import isfile, join
path_tfrecord =  sorted([
                            path + '/' + f 
                            for f in listdir(path) 
                            if isfile(join(path, f))
                        ])


# In[16]:


len(path_tfrecord)


# In[17]:


# # construt train test split
# # 5-folds ==> 80% training / 20% testing

# # train and test sets will be stored so that each experiment may be run separately
# # as opposed to performing model training (lengthy) for all 5 splits in one go 

# train_sets = []
# test_sets  = []

# kf = KFold(n_splits=5, random_state=None, shuffle=True)
# for train_index, test_index in kf.split(path_tfrecord):
    
#     train_set = np.asarray(path_tfrecord)[train_index]
#     test_set  = np.asarray(path_tfrecord)[test_index]
    
#     train_sets.append(train_set)
#     test_sets.append(test_set)
    
# # dump splits to file (so that testing and training are consistent)

# np.asarray(train_sets).dump('./train_sets.pkl')
# np.asarray(test_sets).dump('./test_sets.pkl')


# In[18]:


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


# In[19]:


# will create a tf.dataset for optimized consumption of training data
# the returned image, label tuple represents a call to iterator.get_next()

def create_dataset(filepath, epochs):
    
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)
    
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(parse, num_parallel_calls=8)
    
    # This dataset will go on forever
    dataset = dataset.repeat()
    
    # Set the number of datapoints you want to load and shuffle 
    dataset = dataset.shuffle(100)
    
    # Set the batchsize
    dataset = dataset.batch(2)
    
    # Create an iterator
    iterator = dataset.make_one_shot_iterator()
    
    # Create your tf representation of the iterator
    image, label = iterator.get_next()

    # Bring your picture back in shape
    image = tf.reshape(image, [-1, 128, 128, 1])
    
    # Create a one hot array for your labels
    #label = tf.one_hot(label, 5)
    label = tf.reshape(label, [-1,5])

    
    return image, label


# In[20]:


# hyperparameter init

rows       = 128
cols       = 128
model_name = 'best_psy_cnn'
epochs     = 20  # models have been shown to achieve approx 0.8 acc within 20 epochs
keep_prob  = 0.2 # dropout probability -> tweakable hyperparam 

# create a train dataset with this current train split
image, label = create_dataset(path_tfrecord, epochs)

input_shape = (rows, cols)

# init the input to Keras model as the tf.Dataset iterator
inputs = Input(tensor = image)

# model init
x = BatchNormalization()(inputs)
x = Conv2D(256, kernel_size=(4, cols), activation='relu', input_shape=input_shape)(x)
shortcut = x

# sizes of convolutional layers, filter window size, activation functions
x = Conv2D(256, kernel_size=(4, 1), activation='relu', padding='same')(x)
x = Conv2D(256, kernel_size=(4, 1), activation='relu', padding='same')(x)
x = Conv2D(256, kernel_size=(4, 1), activation='relu', padding='same')(x)
x = Conv2D(512, kernel_size=(4, 1), activation='relu', padding='same')(x)

# usage of both average + max pooling, or either 
x1 = AveragePooling2D(pool_size=(125, 1))(keras.layers.concatenate([x, shortcut]))

x2 = MaxPooling2D(pool_size=(125, 1))(keras.layers.concatenate([x, shortcut]))

x = Dropout(keep_prob)(keras.layers.concatenate([x1, x2]))

x = Dropout(keep_prob)(x)

x = Flatten()(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(keep_prob)(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(keep_prob)(x)

pred = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=inputs, outputs=pred)


# In[21]:


# compile the keras model
# usage of Adam, AdaDelta, RMSProp, Stochastic Gradient Descent
model.compile(loss           = keras.losses.categorical_crossentropy,
              optimizer      = keras.optimizers.Adam(),
              metrics        = ['acc'],
              target_tensors = [label])


# In[22]:


# init checkpoints of model
checkpoint     = ModelCheckpoint(model_name+'.h5', 
                                 monitor        = 'acc', 
                                 verbose        = 1, 
                                 save_best_only = True, 
                                 mode           = 'max')


# In[23]:


# init early stopping
early_stop     = EarlyStopping(monitor='acc', 
                               patience=5, 
                               mode='max') 


# In[24]:


# init tensorboard callbacks (graph visualization)
tensorboard    = keras.callbacks.TensorBoard(log_dir                = './logs', 
                                             histogram_freq         = 0, 
                                             batch_size             = 32, 
                                             write_graph            = True, 
                                             write_grads            = False, 
                                             write_images           = True, 
                                             embeddings_freq        = 0, 
                                             embeddings_layer_names = None, 
                                             embeddings_metadata    = None, 
                                             embeddings_data        = None, 
                                             update_freq            = 'epoch')


# In[25]:


# Thanks to Marcin Możejko - https://stackoverflow.com/a/43186440

# init a Time Callback to record computation time per epoch metrics

class TimeHistory(keras.callbacks.Callback):
        
    def on_train_begin(self, logs={}):
        self.times = []
        self.times_per_step = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.process_time()

    def on_epoch_end(self, batch, logs={}):
        
        steps_per_epoch = 92
        
        result = time.process_time() - self.epoch_time_start
        self.times.append(result)
        self.times_per_step.append(result / steps_per_epoch)
        
time_callback = TimeHistory()


# In[26]:


callbacks_list = [checkpoint, 
                  early_stop, 
                  tensorboard,
                  time_callback]


# In[29]:


expt_name = str(model.optimizer).split('.')[2].split(' ')[0]
expt_name = str(model.layers[8]).split('.')[3].split(' ')[0]
expt_name


# In[30]:


# train model with current hyperparameters 

# steps per epoch = 92 since there 184 TFRecords (each with 150 shuffled spectrograms-
# 30 of each genre), and the iterator retrieves batches of 2 TFRecords
# So for each epoch, all 184 TFRecords are consumed 

history = model.fit(                            batch_size       = None,
                            epochs           = epochs,
                            verbose          = 1,
                            shuffle          = True,
                            callbacks        = callbacks_list,
                            steps_per_epoch  = 92
                    )


# In[19]:


times = time_callback.times
times_per_step = time_callback.times_per_step
acc = history.history['acc']
epoch_col = [i for i in range(1,21)]


# In[20]:


# init result dataframe

result_df = pd.DataFrame(columns = ['epoch', 'acc', 'time'])
result_df['epoch'] = epoch_col
result_df['acc']   = acc
result_df['time']  = times


# In[21]:


# save experiment training results to csv

expt_folder_name = 'pool/'
results_name     = ('pool_' + 
                    str(expt_name).replace('.','') + 
                    'fold_' + 
                    str(train_test_index+1) + 
                    '.csv')
results_path     = './results/' + expt_folder_name + results_name

print(results_path)

result_df.to_csv(results_path)


# In[22]:


len(path_tfrecord)


# In[23]:


model.summary()


# In[ ]:









































































