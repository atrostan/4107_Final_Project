#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import librosa
import librosa.display
import numpy as np
import pickle
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import IPython.display as ipd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


# In[2]:


integer_encoded = np.array([0,1,2,3,4])
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


# In[3]:


onehot_encoded


# In[4]:


dir_path = os.path.dirname(os.path.realpath('./'))
dir_path


# In[5]:


# thanks to https://stackoverflow.com/a/4602224/8822734
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# In[6]:


# find all mp3 files in your root dir
def find_mp3s(rootdir):
    mp3s = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            print(file)
            curr_path = str(os.path.join(subdir, file))
            if(curr_path.endswith('mp3')):
                mp3s.append(curr_path)
    return mp3s


# In[7]:


# one hot encode all 5 labels in experiment 
def ohe(label):
    if label == 0: # dark
        return np.array([1., 0., 0., 0., 0.])
    elif label == 1: # downtempo
        return np.array([0., 1., 0., 0., 0.])
    elif label == 2: # fullon
        return np.array([0., 0., 1., 0., 0.])
    elif label == 3: # goa
        return np.array([0., 0., 0., 1., 0.])
    elif label == 4: # techno
        return np.array([0., 0., 0., 0., 1.])


# In[8]:


# different utility functions used in the experiment

def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# only wrap_floats was needed since 
# x: spectrograms were stored as float np.arrays of 128x128
# y: labels were stored as float np.arrays of 1x5
def wrap_floats(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

def wrap_string(value):
    return tf.train.Feature(string_list = tf.train.Str)


# In[9]:


def serialize_slices(tens_arr, lab_arr, tid):
    
    out_path = '/home/alextrosta/Data/4107/Final/tf_records/'
    
    out_path = out_path + str(tid) + '.tfrecord'
    

    with tf.python_io.TFRecordWriter(out_path) as writer:


        # Create a dict with the data we want to save in the
        # TFRecords file. You can add more relevant data here.
        data =             {
                'image': wrap_floats(tens_arr),
                'label': wrap_floats(lab_arr)
            }

        # Wrap the data as TensorFlow Features.
        feature = tf.train.Features(feature=data)

        # Wrap again as a TensorFlow Example.
        example = tf.train.Example(features=feature)

        # Serialize the data.
        serialized = example.SerializeToString()

        # Write the serialized data to the TFRecords file.
        writer.write(serialized)


# In[10]:


def compute_log_spectrogram(y, sr):
    D = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D)

    # Passing through arguments to the Mel filters
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                        fmax=12000)
    log_mel = (librosa.core.amplitude_to_db(S))
    print(log_mel.shape)
    spec_split = window_partition(log_mel)
    #librosa.display.specshow(spec)
    return spec_split


# In[11]:


def norm_spec_split(spec_split):
    normed = []
    for s in spec_split:
        scaler = MinMaxScaler()
        scaler.fit(s)
        s_normed = scaler.transform(s)
        normed.append(s_normed)
    
    return np.array(normed)


# In[12]:


def mp3_album_names(dk_mp3s, dt_mp3s, fo_mp3s, go_mp3s, te_mp3s):
    dk_al = set                    (
                        [f.replace(mp3dir, '').replace('/dark/','').split('/')[0] 
                         for f in sorted(dk_mp3s)]
                    )
    dt_al = set                    (
                        [f.replace(mp3dir, '').replace('/downtempo/','').split('/')[0] 
                         for f in sorted(dt_mp3s)]
                    )
    fo_al = set                    (
                        [f.replace(mp3dir, '').replace('/fullon/','').split('/')[0] 
                         for f in sorted(fo_mp3s)]
                    )
    go_al = set                    (
                        [f.replace(mp3dir, '').replace('/goa/','').split('/')[0] 
                         for f in sorted(go_mp3s)]
                    )
    te_al = set                    (
                        [f.replace(mp3dir, '').replace('/techno/','').split('/')[0] 
                         for f in sorted(te_mp3s)]
                    )
    return dk_al, dt_al, fo_al, go_al, te_al


# In[13]:


def build_url(album_string):
    split = album_string.split('-')
    stripped = [el.lower().replace('.', '').strip().replace(' ', '-') for el in split]
    url = stripped[0] + '-' + stripped[1]
    url = 'www.ektoplazm.com/free-music/' + url
    return url


# In[14]:


def mp3_album_urls(dk_mp3s, dt_mp3s, fo_mp3s, go_mp3s, te_mp3s):
    dk_al, dt_al, fo_al, go_al, te_al =  mp3_album_names                                                        (
                                                            dk_mp3s, 
                                                            dt_mp3s, 
                                                            fo_mp3s, 
                                                            go_mp3s, 
                                                            te_mp3s
                                                        )
    
    dk_url = [build_url(a) for a in dk_al]
    dt_url = [build_url(a) for a in dt_al]
    fo_url = [build_url(a) for a in fo_al]
    go_url = [build_url(a) for a in go_al]
    te_url = [build_url(a) for a in te_al]
    
    return dk_url, dt_url, fo_url, go_url, te_url


# In[15]:


# your mp3 directory goes here!
mp3dir = '/home/alextrosta/Data/mp3s/mp3'

rootdir = mp3dir + '/dark/'
dk_mp3s = find_mp3s(rootdir)

np.random.shuffle(dk_mp3s)

rootdir = mp3dir + '/downtempo/'
dt_mp3s = find_mp3s(rootdir)

np.random.shuffle(dt_mp3s)

rootdir = mp3dir + '/fullon/'
fo_mp3s = find_mp3s(rootdir)

np.random.shuffle(fo_mp3s)

rootdir = mp3dir + '/goa/'
go_mp3s = find_mp3s(rootdir)

np.random.shuffle(go_mp3s)

rootdir = mp3dir + '/techno/'
te_mp3s = find_mp3s(rootdir)

np.random.shuffle(te_mp3s)


# In[16]:


rootdir


# In[17]:


dk_url, dt_url, fo_url, go_url, te_url = mp3_album_urls(dk_mp3s, 
                                                           dt_mp3s, 
                                                           fo_mp3s, 
                                                           go_mp3s, 
                                                           te_mp3s)


# In[23]:


te_url


# In[ ]:


length = min(len(dk_mp3s), 
             len(dt_mp3s),
             len(fo_mp3s),
             len(go_mp3s),
             len(te_mp3s))

print(length)

d = []

for i in range(length):
    
    d.append({
        'filename' : dk_mp3s[i],
        'label'    : 0
    })
    
    d.append({
        'filename' : dt_mp3s[i],
        'label'    : 1
    })
    
    d.append({
        'filename' : fo_mp3s[i],
        'label'    : 2
    })
    
    d.append({
        'filename' : go_mp3s[i],
        'label'    : 3
    })
    
    d.append({
        'filename' : te_mp3s[i],
        'label'    : 4
    })
    
    print(i)


# In[ ]:


did = 0

import gc

for d in ds:
    slice_arr = []
    tens_arr  = []
    lab_arr   = []
    
    gc.collect()
    tid = 0
    did += 1
    
    for s in d:
        
        gc.collect()
        fn    = s['filename']
        label = s['label']

        print('reading ' + fn)
        print('its label = ' + str(label))

        # read file
        y, sr      = librosa.core.load(fn ,sr = 44100,mono = True)

        # create mel-log spectrogram
        spec_split = np.array(compute_log_spectrogram(y, sr))

        # sample 30 slices from middle of song
        start = len(spec_split) // 2 - 15
        end   = len(spec_split) // 2 + 15

        print(start)
        print(end)

        # normalize spectrogram
        normed     = norm_spec_split(spec_split[start:end])

        print(normed.shape)

        # make a label array that corresponds to each slice of the spectrogram

        length = normed.shape[0]
        label_rep = np.tile(ohe(label), (length,1))

        print(label_rep.shape)

        tens_arr.append(normed)
        lab_arr.append(label_rep)

        # if we've processed one of each genre, export to TFRecord
        if len(tens_arr) % 5 == 0:

            lab_arr  = np.vstack((np.array(lab_arr[0]), 
                                  np.array(lab_arr[1]),
                                  np.array(lab_arr[2]),
                                  np.array(lab_arr[3]),
                                  np.array(lab_arr[4])))

            tens_arr = np.vstack((np.array(tens_arr[0]), 
                                  np.array(tens_arr[1]),
                                  np.array(tens_arr[2]),
                                  np.array(tens_arr[3]),
                                  np.array(tens_arr[4])))

            # shuffle data and labels (in unison)
            tens_arr, lab_arr = unison_shuffled_copies(tens_arr, lab_arr)

            # serialize and save to TFRecord file
            
            final_id = str(str(did)+'_'+str(tid))
            
            serialize_slices(tens_arr, lab_arr, final_id)
            
            tens_arr = []
            lab_arr = []
            tid += 1
        

# stack both normed spectrogram arrays together (1 from each class)


# In[ ]:
















































































