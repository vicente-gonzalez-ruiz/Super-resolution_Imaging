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
    test_photos = [f for f in os.listdir(test_dir) if os.path.isfile(test_dir + f)]

    print("--------------------------------------------------------------------------------")

    for photo in test_photos:
        
        #---
        img_open = Image.open(test_dir + photo)
        width_pc , height_pc = img_open.size
        img_new = img_open.resize((width_pc*2, height_pc*2), Image.ANTIALIAS)
        #img_new.save(test_dir + photo)
        #---

        # load training image and crop it if necessary

        print("Processing image " + photo)
        #image = np.float16(misc.imresize(misc.imread(test_dir + photo), res_sizes[phone])) / 255
        image = np.float16(misc.imresize(img_new, res_sizes[phone])) / 255


        image_crop = utils.extract_crop(image, resolution, phone, res_sizes)
        image_crop_2d = np.reshape(image_crop, [1, IMAGE_SIZE])

        # get enhanced image

        enhanced_2d = sess.run(enhanced, feed_dict={x_: image_crop_2d})
        enhanced_image = np.reshape(enhanced_2d, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

        #before_after = np.hstack((image_crop, enhanced_image))
        photo_name = photo.rsplit(".", 1)[0]

        # save the results as .png images

        misc.imsave("output/" + photo_name + "_original_2X.png", image_crop)
        misc.imsave("output/" + photo_name + "_processed_2X.png", enhanced_image)
        #misc.imsave("results/" + photo_name + "_before_after.png", before_after)
