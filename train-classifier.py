#!/usr/bin/env python3

import os
import signal
import zipfile
import datetime

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

# use the InceptionV3 pre-trained model for feature extraction
# since we're only using it for feature extraction, we don't need
# to train this model
# Our input feature map is 240x240x3: 240x240 for the image pixels, and 3 for
# the three color channels: R, G, and B
local_weights_file = 'inception_v3_weights.h5'
# NB you can acquire this file with the following commands:
# wget https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
# mv inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 inception_v3_weights.h5
pre_trained_model = InceptionV3(
    input_shape = (240, 240, 3),
    include_top = False,
    weights = None
)
pre_trained_model.load_weights(local_weights_file)
for layer in pre_trained_model.layers:
    layer.trainable = False
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# add another fully connected layer with 128 hidden units and ReLU activation
x = layers.Dense(128, activation='relu')(x)
# Add a dropout rate of 0.3
x = layers.Dropout(0.3)(x)
# Add a final softmax layer with 5 nodes for classification
x = layers.Dense(5, activation = 'softmax')(x)

# Configure and compile the model
model = Model(pre_trained_model.input, x)
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = RMSprop(lr = 0.0001),
    metrics = ['acc']
)

# define example directories & files
base_dir = './flowers-scaled/'
train_dir = os.path.join(base_dir, 'training')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'testing')

"""
# training directories for flowers
train_daisy = os.path.join(train_dir, 'daisy')
train_dandelion = os.path.join(train_dir, 'dandelion')
train_rose = os.path.join(train_dir, 'rose')
train_sunflower = os.path.join(train_dir, 'sunflower')
train_tulip = os.path.join(train_dir, 'tulip')

# validation directories for flowers
validation_daisy = os.path.join(validation_dir, 'daisy')
validation_dandelion = os.path.join(validation_dir, 'dandelion')
validation_rose = os.path.join(validation_dir, 'rose')
validation_sunflower = os.path.join(validation_dir, 'sunflower')
validation_tulip = os.path.join(validation_dir, 'tulip')

# testing directories for flowers
test_daisy = os.path.join(test_dir, 'daisy')
test_dandelion = os.path.join(test_dir, 'dandelion')
test_rose = os.path.join(test_dir, 'rose')
test_sunflower = os.path.join(test_dir, 'sunflower')
test_tulip = os.path.join(test_dir, 'tulip')
"""

# add data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

# note that the validation data should not be augmented
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, # This is the source directory for training images
    target_size = (240, 240),  # All images will be resized to 240x240
    batch_size = 20,
    # Since we use categorical_crossentropy loss, we need categorical labels
    class_mode = 'categorical'
)

# flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size = (240, 240),
    batch_size = 20,
    class_mode = 'categorical'
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 2,
    validation_data = validation_generator,
    validation_steps = 50,
    verbose = 2
)

# NB this should only be done AFTER we have pre-trained the model
# fine-tune the higher layers (after 'mixed6') of the pre-trained model to be more
# specific to this flowers dataset
unfreeze = False
for layer in pre_trained_model.layers:
    if unfreeze:
        layer.trainable = True
    if layer.name == 'mixed6':
        unfreeze = True

# As an optimizer, here we will use SGD
# with a very low learning rate (0.00001)
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = SGD(
        lr = 0.00001,
        momentum = 0.9
    ),
    metrics = ['acc']
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_generator,
    validation_steps = 50,
    verbose = 2
)

# TODO make examine metrics for model performance
"""
# plot training & validation accuracy & loss per epoch
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
"""

# TODO generate & examine a confusion matrix
"""
#Print confusion matrix of all items in validation set
evaluation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(240, 240),
    batch_size=1,
    class_mode='categorical',
    shuffle=False)

y_pred_frac = model.predict_generator(evaluation_generator, evaluation_generator.classes.size)
y_pred = np.argmax(y_pred_frac, axis=1)
print('Confusion Matrix')
print(confusion_matrix(evaluation_generator.classes, y_pred))
"""

# save the model to an HDF5 file
output_dir = 'weights'
output_file = os.path.join(output_dir, 'flower-model ' + str(datetime.datetime.now()).split('.')[0] + '.h5')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model.save(output_file)
print("Saved model as %s" % output_file)
