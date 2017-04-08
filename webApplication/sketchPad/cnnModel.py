from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import pandas as pd
from scipy import stats

import re
import base64
from PIL import Image
from io import BytesIO
import base64

def preprocess(data):
    """
        Function takes a base64 encode image data and returns an image array that can be passed into a Keras model
    """
    # dimensions of our images.
    img_width, img_height = 250, 250
    dataUrlPattern = re.compile('data:image/(png|jpeg);base64,(.*)$')
    imgb64 = dataUrlPattern.match(data).group(2)
    if imgb64 is not None and len(imgb64) > 0:
        data= base64.b64decode(imgb64)
        im1 = Image.open(BytesIO(data))
        im1=im1.convert('RGB')
    im1= im1.resize((img_width,img_height))
    print("[INFO] loading and preprocessing image...")
    image = img_to_array(im1)
    image = image.reshape((1,) + image.shape)  # this is a Numpy array with shape (1, 3, 250,250)
    return image

def build_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, img_width, img_height)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(250))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.load_weights('bestWeight.hdf5')
    return model

def load_labels():
    df = pd.read_csv('labels.csv',
        header=0)
    target_names = df['Category'].tolist()
    return target_names

def predict_labels(data):
    model = build_model()
    image = preprocess(data)
    target_names = load_labels()
    encoder = LabelEncoder()
    encoder.fit(target_names)
    pL = model.predict(image)
    prob = model.predict_proba(image)

    p= np.argsort(pL, axis=1)
    n1 = (p[:,-4:]) #gives top 5 probabilities
    pL_names = (encoder.inverse_transform(n1))
    pL_names = pL_names[0]

    p= np.sort(prob, axis=1)
    n = (p[:,-4:]) #gives top 5 probabilities
    prob_values = [stats.percentileofscore(n[0], a, 'mean') for a in n[0]]
    return zip(pL_names,prob_values)
