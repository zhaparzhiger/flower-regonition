#!/usr/bin/env python3

import os, sys
from socket import *

# connect to server
def connect():
    addr = '127.0.0.1'
    port = 14000
    sock = socket(AF_INET, SOCK_STREAM)
    sock.connect((addr, port))
    return sock

if __name__ == "__main__":
    # usage ./classify-client.py test-images/
    # TODO rework this to use ArgumentParser
    # TODO get filenames to examine

    # check command-line arguments
    if len(sys.argv) < 2:
        print("usage: ./classify-client.py test-images/")
        sys.exit(2)
    image_dir = sys.argv[1]
    if not os.path.isdir(image_dir):
        print("usage: ./classify-client.py test-images/")
        sys.exit(2)

    # setup connection
    sock = connect()
    sock.send(bytes(image_dir, 'utf-8')) # send image directory

    # receive predictions from server
    predictions = sock.recv(4096)
    predict_str = str(predictions, 'utf-8')

    # split predictions & print
    for prediction in predict_str.split(","):
        print(prediction)

    # close connection
    sock.close()
