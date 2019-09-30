# To test the model use the following code:
# python run_model.py resolution=<resolution> use_gpu=<use_gpu>
# <resolution> = {orig, high, medium, small, tiny}
# <use_gpu> = {true, false}

from scipy import misc
import numpy as np
import tensorflow as tf
from model import resnet
import utils
import os
import sys
from PIL import Image
import pywt
import cv2


# process command arguments
phone,resolution, use_gpu = utils.process_command_args(sys.argv)

# get all available image resolutions
res_sizes = utils.get_resolutions()

# get the specified image resolution
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE = utils.get_specified_res(res_sizes, phone, resolution)

# disable gpu if specified
config = tf.ConfigProto(device_count={'GPU': 0}) if use_gpu == "false" else None

# create placeholders for input images
x_ = tf.compat.v1.placeholder(tf.float32, [None, IMAGE_SIZE])
x_image = tf.reshape(x_, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

# generate enhanced image
enhanced = resnet(x_image)

with tf.compat.v1.Session(config=config) as sess:

    # load pre-trained model
    saver = tf.train.Saver()
    saver.restore(sess, "models_orig/" + phone)

    test_dir = "input/"
    print_dir = "output/"
    test_photos = [f for f in os.listdir(test_dir) if os.path.isfile(test_dir + f)]

    print("--------------------------------------------------------------------------------")

    for photo in test_photos:
        
        #---
        img_open = Image.open(test_dir + photo)
        width_pc , height_pc = img_open.size
        img_new = img_open.resize((width_pc*2, height_pc*2), Image.ANTIALIAS)
        #img_new.save(print_dir + photo)
        #---

        imgc = cv2.imread(test_dir + photo)
        imgc = cv2.resize(imgc, ( width_pc-4, height_pc-4) )
        imgc = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
        
        # Wavelet transform of image, and plot approximation and details
        coeff = pywt.dwt2(imgc, 'bior1.3')
        LL, (LH,HL,HH) = coeff

        
        # load training image and crop it if necessary

        print("Processing image " + photo)
        #image = np.float16(misc.imresize(misc.imread(test_dir + photo), res_sizes[phone])) / 255
        img_orig = np.float16(misc.imresize(img_new, [width_pc, height_pc])) /255
        image = np.float16(misc.imresize(img_new, res_sizes[phone])) / 255


        image_crop = utils.extract_crop(image, resolution, phone, res_sizes)
        image_crop_2d = np.reshape(image_crop, [1, IMAGE_SIZE])

        # get enhanced image

        enhanced_2d = sess.run(enhanced, feed_dict={x_: image_crop_2d})
        enhanced_image = np.reshape(enhanced_2d, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        img_open_orig = np.float16( misc.imresize(img_open, [width_pc*2, height_pc*2])) / 255

        #before_after = np.hstack((image_crop, enhanced_image))
        photo_name = photo.rsplit(".", 1)[0]

        for i,a in enumerate([LL, LH, HL, HH]):
        	cv2.imwrite('output/' + photo_name + '_pywt_1X.png',a)
        	break

        # save the results as .png images
        misc.imsave("output/" + photo_name + "_original_2X.png", img_orig)
        #misc.imsave("output/" + photo_name + "_original_2X.png", img_open_orig)
        misc.imsave("output/" + photo_name + "_processed_2X.png", enhanced_image)
        #misc.imsave("results/" + photo_name + "_before_after.png", before_after)
