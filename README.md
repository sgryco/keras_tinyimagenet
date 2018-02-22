MirrorNet Keras implementation
=====

This project was written by Corentin Chéron
for the Computer Vision project on Deep Learning.

Main files
----

* `train.py` Main training file. With parameters, allows to run sequence of training or reload existing weights. Depends on:
  * `callbacks.py` All the functions run between two epochs
  * `dataset_loading.py` Related to loading images, creating hdf5 and data augmentation
  * `model.py` Neural networks definitions
  
* `extract_scalars.py` and `csv_stats.py` Batch extraction of variable from the Tensorboard logs into csv and compute summary statistics 
* `predict_test.py` Create a Kaggle submission file from test_data and the name of a run
* `visualize.py` Create a confusion matrix graph and export confusion graph for Gephi

* `create_tiny_image_net_data.sh` Download the Tiny ImageNet dataset to the data folder

* `class_mapping_nid.txt` and `class_mapping.txt` Txt files containing mapping between class names, folder names and indices.


Requirements
---
Tensorflow, Keras, pydot, scikit-image, scikit-learn, numpy, Pillow


Demo
---
Retreive the Tiny ImageNet dataset:
```
./create_tiny_image_net_data.sh
```

Train a demo model:
```
python train.py --name "demo" --model demo_model 
```

Train the MirrorNet-4 network:
```
python train.py --name "MirrorNet-4" --sequence
```

Visualise the curves, architecture and parameters with tensorboard:
```
tensorboard --logdir=tensorboard
```