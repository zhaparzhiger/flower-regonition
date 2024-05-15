#!/usr/bin/env python3

from socket import *

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

# setup tcp server
def tcp_setup():
    addr = '127.0.0.1'
    port = 14000
    sock = socket(AF_INET, SOCK_STREAM)
    sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    sock.bind((addr, port))
    sock.listen(1)
    return sock

def build_predict_str(file, predictions):
    list = [ "Predictions for '%s'\n" % file ]
    for i in range(0, len(predictions)):
        list.append("\t%s: %.5f\n" % (labels[i], predictions[i]))
    return "".join(list)

if __name__ == "__main__":
    # usage: ./classify-server.py "weights/flower-model YYYY-MM-DD HH:MM:SS.h5"
    # TODO rework this to use ArgumentParser

    # check command-line arguments
    if len(sys.argv) < 2:
        print("usage: ./classify-server.py \"weights/flower-model YYYY-MM-DD HH:MM:SS.h5\"")
        sys.exit(2)
    model_file = sys.argv[1]
    if not model_file.endswith(".h5") or not os.path.exists(model_file):
        print("usage: ./classify-server.py \"weights/flower-model YYYY-MM-DD HH:MM:SS.h5\"")
        sys.exit(2)

    # load the model & get set up for predictions
    # if the saved model is invalid, alert the user and abort
    model = load_the_model(model_file)

    # setup socket
    sock = tcp_setup()

    # listen for connections until SIGTERM is received
    print("Server started. Use ^C to exit.")

    while True:
        # establish connection with client
        connection = sock.accept()[0]
        print("Processing request from: " + str(connection.getpeername()[0]) \
            + "...", end='')

        # receive request
        request = connection.recv(1024)
        image_dir = str(request, 'utf-8') # this is the image directory name

        # get images from directory specified in request
        # load images from the given directory
        # if there are no images, alert the user and abort
        images = os.listdir(image_dir)
        if len(images) == 0:
            print("No images found")
            sys.exit(1)

        # classify each image, skipping subdirectories
        predict_list = []
        for item in images:
            if os.path.isdir(os.path.join(image_dir, item)):
                continue
            else:
                image_path = os.path.join(image_dir, item)
                predictions = predict(image_path, model)
                predict_list.append(build_predict_str(image_path, predictions))

        # build response with predictions
        response_str = ",".join(predict_list)

        # send response to client
        response = bytes(response_str, 'utf-8')
        connection.send(response)
        print("done.")

        # close the connection
        connection.close()
