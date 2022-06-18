# =========== ====  Convolutional Neural Network : CNN  ==== ============

# ----------- Install following  packages -------------
    # Install Theano
    # Install Tensorflow
    # Install Tensorflow 
    # Install Keras 


# ---------- Part 1 - Building the CNN ----------------
# Importing the Keras libraries and packages 

from keras.models import Sequential     # to initialize as sequence-of-layers
from keras.layers import Convolution2D  # Convolution step for images
from keras.layers import MaxPooling2D   # not "MaxPool2D". Pooling step for images
from keras.layers import Flatten        # Flatenning step
from keras.layers import Dense          # ads fully-connected-layers to classic ANN

# initializing the CNN
cnn_classifier = Sequential()

# step 1 : Convolution - layer
cnn_classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation= "relu"))

# step 2 : Pooling  - layer
cnn_classifier.add(MaxPooling2D(pool_size = (2, 2)))

# improving step : Adding 2nd-Convolution and  2nd-Pooling layers
cnn_classifier.add(Convolution2D(32, 3, 3, activation= "relu"))
cnn_classifier.add(MaxPooling2D(pool_size = (2, 2)))

# step 3 : Flattening 
cnn_classifier.add(Flatten())

# step 4 : ANN - full connection
cnn_classifier.add(Dense(units= 128, activation= "relu")) # fully connected layers
cnn_classifier.add(Dense(units= 1, activation= "sigmoid")) # output layer

# compile the NN
cnn_classifier.compile(optimizer= "adam", loss="binary_crossentropy", metrics = ["accuracy"])


# ---------- Part 2 - Image Preprocessing & fit CNN to our images ----------------

from keras.preprocessing.image import ImageDataGenerator
import math

# creating two data-generator/augmentation obgects for train and test data
        # here we specify the transform parameters

train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True )

test_datagen = ImageDataGenerator(rescale=1./255)

# applying augmentation on training data : training folder path needed
                                        # target_size = input_shape (dimension of expected resized images)
                                        # batch_size is not related to no. of filters
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

# applying augmentation on test data : test folder path needed
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

"""
cnn_classifier.fit_generator(training_set,
 # due to incompatibility between Tensorflow and Keras version, "samples_per_epoch", "nb_epoch", "nb_val_samples" may not work 
 # Then use "steps_per_epoch", "epochs",  "validation_steps" instead
                            # samples_per_epoch = 8000,
                            # nb_epoch=25,
                            # nb_val_samples = 2000,
                            steps_per_epoch = 250,  # training_set_size/batch_size
                            epochs = 25,
                            validation_data=test_set,
                            validation_steps = 62   # test_set_size/batch_size
                            )

"""


# In this case no need to explicitly specify training_set's or test_set's sample-size: i.e 8000, 2000 or 800, 200
cnn_classifier.fit_generator(     
                            training_set,
                            steps_per_epoch=math.floor((training_set.samples)/(training_set.batch_size)),
                            epochs=25,
                            validation_data=test_set,
                            validation_steps=math.floor((test_set.samples)/(test_set.batch_size))
                            )

# history = model.fit_generator(train_gen,
#                                   steps_per_epoch=(train_gen.samples/batch_size),  # len(train_gen)
#                                   epochs=100,
#                                   validation_data=validation_gen,
#                                   validation_steps=(validation_gen.samples/batch_size),
#                                   callbacks=[checkpointer],
#                                   workers=4
#                                   )

# python prctc_cnn.py
