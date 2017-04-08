from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.template import Context, loader
from django.http import JsonResponse
from django.template import RequestContext
from django.utils.datastructures import MultiValueDictKeyError
import json
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
import os
import re
import base64
from PIL import Image
from io import BytesIO
import base64

def alpha_to_color(image, color=(255, 255, 255)):
    """Set all fully transparent pixels of an RGBA image to the specified color.
    This is a very simple solution that might leave over some ugly edges, due
    to semi-transparent areas. You should use alpha_composite_with color instead.

    Source: http://stackoverflow.com/a/9166671/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    x = np.array(image)
    r, g, b, a = np.rollaxis(x, axis=-1)
    r[a == 0] = color[0]
    g[a == 0] = color[1]
    b[a == 0] = color[2]
    x = np.dstack([r, g, b, a])
    return Image.fromarray(x, 'RGBA')

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
        im1 = alpha_to_color(im1)
        im1=im1.convert('RGB')
    im1= im1.resize((250,250))
    print("[INFO] loading and preprocessing image...")
    image = img_to_array(im1)
    image = image.reshape((1,) + image.shape)  # this is a Numpy array with shape (1, 3, 250,250)
    test_ob = ImageDataGenerator(rescale=1./255)
    X=[]
    for batch in test_ob.flow(image, batch_size=1):
        X= batch
        break
    return X

def build_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, 250, 250)))
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
    module_dir = os.path.dirname(__file__)  # get current directory
    file_path = os.path.join(module_dir, 'bestWeight.hdf5')
    model.load_weights(file_path)
    return model



def load_labels():
    module_dir = os.path.dirname(__file__)  # get current directory
    file_path = os.path.join(module_dir, 'labels.csv')
    df = pd.read_csv(file_path,
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
    n1 = (p[:,-4:]) #gives top 5 labels
    pL_names = (encoder.inverse_transform(n1))
    pL_names = pL_names[0]

    p= np.sort(prob, axis=1)
    convertperc = [stats.percentileofscore(p[0], a, 'rank') for a in p[0]]
    n = (convertperc[-4:]) #gives top 5 probabilities perc
    prob_values = (p[:,-4:])
    prob_single_values = prob_values[0]
    return zip(pL_names,n,prob_single_values)

@csrf_exempt
def index(request):
    print ('hi')
    return render(request,"sketchPad/index.html", {})

@csrf_exempt
def recognizeSketch(request):
    context = RequestContext(request)
    if request.method == 'GET':
        return render(request,"sketchPad/index.html", {})
    else:
        data = request.POST['image']
        try:
            print ("found 1")
            data = request.POST['image']
            result = predict_labels(data)
            l1 = result[3][0]
            l2= result[2][0]
            l3 = result[1][0]
            l4= result[0][0]
            p1 = result[3][1]
            p2 = result[2][1]
            p3= result[1][1]
            p4 = result[0][1]

            pr1 = result[3][2]
            pr2 = result[2][2]
            pr3= result[1][2]
            pr4 = result[0][2]

            print ("done running model ")
            data = '<h2>Object recognised as <span style="color:green;">'+l1+'</span></h2><p>Top four categories for the above skecth:</p><div class="progress"><div class="progress-bar progress-bar-success" role="progressbar" aria-valuenow="'+str(p1)+'" aria-valuemin="0" aria-valuemax="100" style="width:' +str(p1)+'%">'+str(pr1)+' '+l1+'</div></div><div class="progress"><div class="progress-bar progress-bar-info" role="progressbar" aria-valuenow="'+str(p2)+'" aria-valuemin="0" aria-valuemax="100" style="width:'+str(p2)+'%">'+str(pr2)+' '+l2+'</div></div><div class="progress"><div class="progress-bar progress-bar-warning" role="progressbar" aria-valuenow="'+str(p3)+'" aria-valuemin="0" aria-valuemax="100" style="width:'+str(p3)+'%">'
            s1 = str(pr3)+' '+l3+'</div></div><div class="progress"><div class="progress-bar progress-bar-danger" role="progressbar" aria-valuenow="'+str(p4)+'" aria-valuemin="0" aria-valuemax="100" style="width:'+str(p4)+'%">'+str(pr4)
            s =  ' '+ l4 + '</div></div>'
            data = data + s1 + s
            return HttpResponse(data)
        except MultiValueDictKeyError:
            return render(request,"sketchPad/index.html", {})
