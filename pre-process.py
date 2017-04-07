
# coding: utf-8

# In[1]:

#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.models import model_from_json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from PIL import Image
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
filepath="/python/sketchANetModel/weights-TestNew-{epoch:02d}-{val_acc:.2f}.hdf5"
# filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# In[2]:

# input image dimensions
img_rows, img_cols = 500, 500

# number of channels
img_channels = 1

#%%
#  data

path1 = '/python/png'    #path of folder of images
path2 = '/python/input_data_resized'  #path of folder to save images

categories = os.listdir(path1)
#
for l in categories:
    filePath = path1 +'/'+ l
    listing = os.listdir(filePath)
    for file in listing:
        im = Image.open(filePath + '/' + file)
        img = im.resize((img_rows,img_cols))
                #need to do some more processing here
        img.save(path2 +'/' +  file, "PNG")

imlist = os.listdir(path)

im1 = array(Image.open(path2 + '/'+ imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images
num_samples = len(imlist)
print imnbr


# In[3]:

# create matrix to store all flattened images
immatrix = array([array(Image.open(path2+ '/' + im2)).flatten()
              for im2 in imlist],'f')
c= 0
label=np.ones((len(imlist),),dtype = int)
for i in range(0,num_samples,80):
    label[i:i+80]= c
    c= c+1
print('Number of classes c')
print(c)

# In[4]:

data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

# img=immatrix[167].reshape(img_rows,img_cols)
# plt.imshow(img)
# plt.imshow(img,cmap='gray')
print('Train data')
print (train_data[0].shape)
print (train_data[1].shape)
print (train_data[1][1])
print (train_data[1][19990])
# In[5]:

#%%
(X, y) = (train_data[0],train_data[1])


# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# In[6]:

#batch_size to train
batch_size = 32
# number of epochs to train
nb_epoch = 200


# In[7]:

#number of classes
nb_classes= c
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)
print('Number of classes from Y_train')
print(Y_test.shape[1])
num_classes = Y_test.shape[1]


i = 100 ## plotting variable
# plt.imshow(X_train[i, 0], interpolation='nearest')
print("label : ", Y_train[i,:])
print("label : ", Y_train[i,:].shape)

# simple cnn
# Create the model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

##
datagen = ImageDataGenerator(
       rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
       width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
       height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
       horizontal_flip=True,  # randomly flip images
       vertical_flip=True)  # randomly flip images

train_generator = datagen.flow(
        X_train, Y_train,  # this is the target directory
        batch_size=batch_size,
        shuffle = True)

validation_datagen = ImageDataGenerator()
validation_generator = validation_datagen.flow(
       X_test, Y_test,  # this is the target directory
       batch_size=batch_size,
       shuffle = True)
# In[10]:
hist = model.fit_generator(train_generator, samples_per_epoch=16000,nb_epoch=nb_epoch,
                           callbacks=callbacks_list,
                           validation_data = validation_generator,
                           nb_val_samples=6)
# hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#              show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
fname = "/sketchANetModel/weights-Preprocess-CNN.hdf5"
model.save_weights(fname,overwrite=True)
