#!/usr/bin/env python3

import os, sys
from random import shuffle, randint

# directory names
TESTDIR = "testing"
VALIDIR = "validation"
TRAINDIR = "training"

# get the second-to-last occurrence of pattern in text
def get_image_filename(image):
    text = os.path.splitext(image)[0]
    pattern = os.sep

    # see https://stackoverflow.com/a/14063233/3646065
    idx =  text.rfind(pattern, 0, text.rfind(pattern))

    return os.path.splitext(image)[0][idx + 1:] + ".jpg"

def make_dirs(rootdir):
    dirs = [TESTDIR, VALIDIR, TRAINDIR]
    for item in dirs:
        dir = os.path.join(rootdir, item)
        if not os.path.exists(dir):
            os.makedirs(dir)
            print("Created directory %s" % dir)

def make_out_dirs(rootdir, subdir):
    dirs = [TESTDIR, VALIDIR, TRAINDIR]
    for outdir in dirs:
        dir = os.path.join(rootdir, outdir, subdir)
        if not os.path.exists(dir):
            os.makedirs(dir)
            print("Created directory %s" % dir)


def make_paths(rootdir, subdir, outdir, image):
    src = os.path.join(rootdir, subdir, image)
    dest = os.path.join(rootdir, outdir, subdir, image)
    return (src, dest)

if __name__ == "__main__":
    # usage: ./separate-images.py flowers-scaled/
    rootdir = sys.argv[1]

    # get image subdirectories
    subdirs = os.listdir(rootdir)

    # make output directories
    make_dirs(rootdir)

    # process images in each subdirectory
    for subdir in subdirs:
        # make output subdirectories
        imagedir = os.path.join(rootdir, subdir)
        # print(imagedir)
        make_out_dirs(rootdir, subdir)

        # randomize image list, then separate into training, validation, and
        # testing groups
        images = os.listdir(imagedir)
        shuffle(images)
        num_images = len(images)

        # print("Processing %s" % imagedir)
        for i in range(0, num_images):
            print("Processing item %d of %d -> " % (i + 1, num_images), end='')
            # print(os.path.join(imagedir, get_image_filename(images[i])))

            # skip directories inside subdirectories
            if os.path.isdir(os.path.join(imagedir, get_image_filename(images[i]))):
                print("skipping directory")
                continue

            # add ~20% of images to 'testing'
            if randint(1, 5) == 1:
                src, dest = make_paths(rootdir, subdir, TESTDIR, get_image_filename(images[i]))
                # print("%s -> %s" % (src, dest))
                print(TESTDIR)
                os.rename(src, dest)
            # add ~16% of images to 'validation'
            elif randint(1, 25) <= 4:
                src, dest = make_paths(rootdir, subdir, VALIDIR, get_image_filename(images[i]))
                # print("%s -> %s" % (src, dest))
                print(VALIDIR)
                os.rename(src, dest)
            # add the remaining ~64% of images to 'training'
            else:
                src, dest = make_paths(rootdir, subdir, TRAINDIR, get_image_filename(images[i]))
                # print("%s -> %s" % (src, dest))
                print(TRAINDIR)
                os.rename(src, dest)

        print("\n") # line break to make output easier to read

        # remove subdir after processing all images
        os.rmdir(imagedir)
