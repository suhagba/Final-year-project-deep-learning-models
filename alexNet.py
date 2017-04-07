from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint

# dimensions of our images.
img_width, img_height = 250, 250

train_data_dir = 'png'
validation_data_dir = 'validation/png'
nb_train_samples = 20000
nb_validation_samples = 800
nb_epoch = 200

# checkpoint
filepath="AlexNET--bestWeight.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

modelA = Sequential()

# Layer 1
modelA.add(Convolution2D(96, 11, 11, input_shape = (3,img_width, img_height), border_mode='same'))
modelA.add(Activation('relu'))
modelA.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
modelA.add(Convolution2D(256, 5, 5, border_mode='same'))
modelA.add(Activation('relu'))
modelA.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
modelA.add(ZeroPadding2D((1,1)))
modelA.add(Convolution2D(512, 3, 3, border_mode='same'))
modelA.add(Activation('relu'))

# Layer 4
modelA.add(ZeroPadding2D((1,1)))
modelA.add(Convolution2D(1024, 3, 3, border_mode='same'))
modelA.add(Activation('relu'))

# Layer 5
modelA.add(ZeroPadding2D((1,1)))
modelA.add(Convolution2D(1024, 3, 3, border_mode='same'))
modelA.add(Activation('relu'))
modelA.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 6
modelA.add(Flatten())
modelA.add(Dense(3072, init='glorot_normal'))
modelA.add(Activation('relu'))
modelA.add(Dropout(0.5))

# Layer 7
modelA.add(Dense(4096, init='glorot_normal'))
modelA.add(Activation('relu'))
modelA.add(Dropout(0.5))

# Layer 8
modelA.add(Dense(250, init='glorot_normal'))
modelA.add(Activation('softmax'))

modelA.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')
import time
start_time = time.time()


history = modelA.fit_generator(
        train_generator,
        callbacks=callbacks_list,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        verbose=1,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)
print("--- %s seconds ---" % (time.time() - start_time))
