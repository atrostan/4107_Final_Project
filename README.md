# 4107_Final_Project

Thanks to:

* https://hackernoon.com/finding-the-genre-of-a-song-with-deep-learning-da8f59a61194
* https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/18_TFRecords_Dataset_API.ipynb
* https://ricardodeazambuja.com/deep_learning/2018/05/04/tuning_tensorflow_docker/
* https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36
* https://github.com/markjay4k/Fashion-MNIST-with-Keras/blob/master/pt3%20-%20FMINST%20Embeddings.ipynb
* https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py

# Requirements:

Seen in [`requirements.txt`](./py/requirements.txt)

|Library|Version|
|-------|-------|
|pandas|0.23.4|
|matplotlib|3.0.2|
|numpy|1.15.4|
|tensorflow|1.12.0|
|scipy|1.1.0|
|ipython|7.2.0|
|keras|2.2.4|
|scikit_learn|0.20.1|

# Download Tracks

Download music tracks from `data_urls.txt`

# Convert Tracks to `TFRecords`

Detailed in `read_mp3s.ipynb`

# Train model 

Detailed in `train_model.ipynb`.  
You can skip this step, and just extract `best_model.tar.gz`: a Keras model
I've trained.   
Load this model using [load_model(best_psy_cnn.h5)](https://stackoverflow.com/a/43263973/8822734)

# Test model 

Detailed in `test_model.ipynb`.  
You can skip this step, and just extract `best_model.tar.gz`: a Keras model
I've trained.   

# Predict using the model

Detailed in `predict_model.ipynb`.   
The model was trained on 5 genres:
* [Downtempo](http://www.ektoplazm.com/style/downtempo)
* [Dark](http://www.ektoplazm.com/style/darkpsy)
* [Techno](http://www.ektoplazm.com/style/techno)
* [Goa](http://www.ektoplazm.com/style/goa)
* [Fullon](http://www.ektoplazm.com/style/full-on)

Download your own psytrance tracks (that can be characterized as one of the
above genres) and see how the model does!

To do this, make sure to compute the [mel-spectrogram](https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html)
of your track, and feed the model with slices of your spectrogram.

# Visualize Convolutional Filters

Detailed in `visualize_filters.ipynb` and 
examples in [`./tex/diagrams/stitched_filters`](./tex/diagrams/stitched_filters)

# Report

The report outlining the development of this project is [here](./tex/report.pdf)

Production of graphs for the report are outlined in `plot_results.ipynb`
