#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

mypath ='/python/png'
mypath2 ='/data'

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

from os import walk
f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from itertools import repeat

categories = os.listdir(mypath)
#
for l in categories:
    filePath = mypath +'/'+ l
    filePath2 = mypath2 +'/'+ l
    if not os.path.exists(filePath2):
	os.makedirs(filePath2)
    os.makedirs(directory)
    datagen = ImageDataGenerator(
         rotation_range=180,
         width_shift_range=0.9,
         height_shift_range=0.2,
         rescale=1./255,
         shear_range=0.9,
         zoom_range=0.9,
         horizontal_flip=True,
         dim_ordering='th')
    c= 1
    for X_batch, y_batch in datagen.flow_from_directory(
        mypath,  # this is the target directory
        target_size=(225,225),  # all images will be resized to 225x225
        batch_size=2,
        shuffle = True, save_to_dir=filePath2, save_prefix='aug', save_format='png'):
        if c>=24000:
            break
    del datagen # delete from memory 
