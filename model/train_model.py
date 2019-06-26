import numpy as np
import sys
from scipy import misc
import tensorflow as tf
import models
import utils
import vgg
from load_dataset import load_test_data, load_batch
from ssim import MultiScaleSSIM

# defining size of image patches

PATCH_WIDTH = 100
PATCH_HEIGHT = 100
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3

# command arguments

phone, batch_size, train_size, learning_rate, num_train_iters, \
w_content, w_color, w_texture, w_tv, \
dped_dir, vgg_dir, eval_step = utils.process_command_args(sys.argv)

# initialize randomness from 0
np.random.seed(0)

# loading training and testing data

print("Loading testing data...")
test_data, test_answ = load_test_data(phone, dped_dir, PATCH_SIZE)

print("Loading training data...")
train_data, train_answ = load_batch(phone, dped_dir, train_size, PATCH_SIZE)

TEST_SIZE = test_data.shape[0]
num_test_batches = int(test_data.shape[0]/batch_size)

# defining system architecture

with tf.Graph().as_default(), tf.Session() as sess:
    
    # placeholders for training data

    phone_ = tf.placeholder(tf.float32, [None, PATCH_SIZE])
    phone_image = tf.reshape(phone_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])

    dslr_ = tf.placeholder(tf.float32, [None, PATCH_SIZE])
    dslr_image = tf.reshape(dslr_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])

    adv_ = tf.placeholder(tf.float32, [None, 1])

    # get processed enhanced image

    enhanced = models.resnet(phone_image)

    # transform both dslr and enhanced images to grayscale

    enhanced_gray = tf.reshape(tf.image.rgb_to_grayscale(enhanced), [-1, PATCH_WIDTH * PATCH_HEIGHT])
    dslr_gray = tf.reshape(tf.image.rgb_to_grayscale(dslr_image),[-1, PATCH_WIDTH * PATCH_HEIGHT])

    # push randomly the enhanced or dslr image to an adversarial CNN-discriminator

    adversarial_ = tf.multiply(enhanced_gray, 1 - adv_) + tf.multiply(dslr_gray, adv_)
    adversarial_image = tf.reshape(adversarial_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 1])

    discrim_predictions = models.adversarial(adversarial_image)

    
    
    for i in range(num_train_iters):

        # train generator

        idx_train = np.random.randint(0, train_size, batch_size)

        phone_images = train_data[idx_train]
        dslr_images = train_answ[idx_train]

        [loss_temp, temp] = sess.run([loss_generator, train_step_gen],
                                        feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: all_zeros})
        train_loss_gen += loss_temp / eval_step

        # train discriminator

        idx_train = np.random.randint(0, train_size, batch_size)

        # generate image swaps (dslr or enhanced) for discriminator
        swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

        phone_images = train_data[idx_train]
        dslr_images = train_answ[idx_train]

        [accuracy_temp, temp] = sess.run([discim_accuracy, train_step_disc],
                                        feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: swaps})
        train_acc_discrim += accuracy_temp / eval_step

        if i % eval_step == 0:

            # test generator and discriminator CNNs

            test_losses_gen = np.zeros((1, 6))
            test_accuracy_disc = 0.0
            loss_ssim = 0.0

            for j in range(num_test_batches):

                be = j * batch_size
                en = (j+1) * batch_size

                swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

                phone_images = test_data[be:en]
                dslr_images = test_answ[be:en]

                [enhanced_crops, accuracy_disc, losses] = sess.run([enhanced, discim_accuracy, \
                                [loss_generator, loss_content, loss_color, loss_texture, loss_tv, loss_psnr]], \
                                feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: swaps})

                test_losses_gen += np.asarray(losses) / num_test_batches
                test_accuracy_disc += accuracy_disc / num_test_batches

                loss_ssim += MultiScaleSSIM(np.reshape(dslr_images * 255, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, 3]),
                                                    enhanced_crops * 255) / num_test_batches

            logs_disc = "step %d, %s | discriminator accuracy | train: %.4g, test: %.4g" % \
                  (i, phone, train_acc_discrim, test_accuracy_disc)

            logs_gen = "generator losses | train: %.4g, test: %.4g | content: %.4g, color: %.4g, texture: %.4g, tv: %.4g | psnr: %.4g, ssim: %.4g\n" % \
                  (train_loss_gen, test_losses_gen[0][0], test_losses_gen[0][1], test_losses_gen[0][2],
                   test_losses_gen[0][3], test_losses_gen[0][4], test_losses_gen[0][5], loss_ssim)

            print(logs_disc)
            print(logs_gen)

           
            # save visual results for several test image crops

            enhanced_crops = sess.run(enhanced, feed_dict={phone_: test_crops, dslr_: dslr_images, adv_: all_zeros})

            idx = 0
            for crop in enhanced_crops:
                before_after = np.hstack((np.reshape(test_crops[idx], [PATCH_HEIGHT, PATCH_WIDTH, 3]), crop))
                misc.imsave('results/' + str(phone)+ "_" + str(idx) + '_iteration_' + str(i) + '.jpg', before_after)
                idx += 1

            train_loss_gen = 0.0
            train_acc_discrim = 0.0

            # save the model that corresponds to the current iteration

            saver.save(sess, 'models/' + str(phone) + '_iteration_' + str(i) + '.ckpt', write_meta_graph=False)

            # reload a different batch of training data

            del train_data
            del train_answ
            train_data, train_answ = load_batch(phone, dped_dir, train_size, PATCH_SIZE)
