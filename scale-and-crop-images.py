#!/usr/bin/env python3

import os, sys
from PIL import Image # docs: http://effbot.org/imagingbook/image.htm

SIZE = 240 # we want a set of 240x240px images
OUT = "flowers-scaled" + os.sep

# scale and crop the given Image to a SIZExSIZE image
def scale_and_crop(img):
    # see https://stackoverflow.com/a/273962/3646065
    width, height = img.size
    aspect_ratio = width / height
    if width < height:
        scaled_img = img.resize((SIZE, int(SIZE / aspect_ratio)), Image.ANTIALIAS)
        cropped_img = scaled_img.crop((0, int((SIZE / aspect_ratio - SIZE) / 2),
            SIZE, SIZE + int((SIZE / aspect_ratio - SIZE) / 2)))
        return cropped_img
    else:
        scaled_img = img.resize((int(SIZE * aspect_ratio), SIZE), Image.ANTIALIAS)
        cropped_img = scaled_img.crop((int((SIZE * aspect_ratio - SIZE) / 2), 0,
            SIZE + int((SIZE * aspect_ratio - SIZE) / 2), SIZE))
        return cropped_img

# get the second-to-last occurrence of pattern in text
def get_second_to_last_substr(text, pattern):
    # see https://stackoverflow.com/a/14063233/3646065
    return text.rfind(pattern, 0, text.rfind(pattern))

# load an image, transform it, and output the result
def process_image(infile):
    # set up output file
    idx = get_second_to_last_substr(os.path.splitext(infile)[0], os.sep)
    outfile = OUT + os.path.splitext(infile)[0][idx + 1:] + "-scaled-and-cropped.jpg"
    print("in: %s\nout: %s" % (infile, outfile))

    # make output directory if it doesn't exist
    idx = os.path.splitext(outfile)[0].rfind(os.sep)
    dir = outfile[:idx]
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Created directory %s" % dir)

    # transform the image
    if infile != outfile:
        try:
            # load image
            img = Image.open(infile)
            # scale & crop
            scaled_and_cropped = scale_and_crop(img)
            # save image
            scaled_and_cropped.save(outfile, "JPEG")
        except IOError as err:
            print("Encountered an error trying to scale and crop %s" % infile)
            print(err)

if __name__ == "__main__":
    # usage: ./scale-and-crop-images.py flowers-raw/
    rootdir = sys.argv[1]
    for infile in os.listdir(rootdir):
        # check for subdirectories
        if os.path.isdir(rootdir + os.sep + infile):
            # process all files in the subdir
            for file in os.listdir(rootdir + os.sep + infile):
                # there are no sub-sub-directories in our dataset
                process_image(os.path.join(rootdir, infile, file))
        else:
            process_image(rootdir + os.sep + infile)
