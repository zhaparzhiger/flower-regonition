#!/usr/bin/env python3

import os, sys

from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input

labels = [ "daisy", "dandelion", "rose", "sunflower", "tulip" ]

def load_the_model(model_file):
    try:
        model = tf.keras.models.load_model(model_file)
        return model
    except ValueError:
        print("Error in '%s' found while loading." % model_file)
        sys.exit(1)

def predict(img_path, model):
    # see https://hackernoon.com/tensorflow-vs-keras-comparison-by-building-a-model-for-image-classification-f007f336c519
    image = Image.open(img_path)
    x = img_to_array(image)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    return predictions.tolist()[0]

def print_result(file, predictions):
    print("Predictions for '%s':" % file)
    for i in range(0, len(predictions)):
        print("\t%s: %.5f" % (labels[i], predictions[i]))

if __name__ == "__main__":
    # usage: ./classifier.py "weights/flower-model YYYY-MM-DD HH:MM:SS.h5" test-images/
    # TODO rework this to use ArgumentParser

    # check command-line arguments
    if len(sys.argv) < 3:
        print("usage: ./classifier.py \"weights/flower-model YYYY-MM-DD HH:MM:SS.h5\" test-images/")
        sys.exit(2)
    model_file = sys.argv[1]
    if not model_file.endswith(".h5") or not os.path.exists(model_file):
        print("usage: ./classifier.py \"weights/flower-model YYYY-MM-DD HH:MM:SS.h5\" test-images/")
        sys.exit(2)
    image_dir = sys.argv[2]
    if not os.path.isdir(image_dir):
        print("usage: ./classifier.py \"weights/flower-model YYYY-MM-DD HH:MM:SS.h5\" test-images/")
        sys.exit(2)

    # load images from the given directory
    # if there are no images, alert the user and abort
    images = os.listdir(image_dir)
    if len(images) == 0:
        print("No images found")
        sys.exit(1)

    # load the model & get set up for predictions
    # if the saved model is invalid, alert the user and abort
    model = load_the_model(model_file)

    # classify each image, skipping subdirectories
    for item in images:
        if os.path.isdir(os.path.join(image_dir, item)):
            continue
        else:
            image_path = os.path.join(image_dir, item)
            predictions = predict(image_path, model)
            print_result(image_path, predictions)
