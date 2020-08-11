#!/usr/bin/env python
"""
  * Test script for <RGB to/from LAB> color-space conversion (tf-1.x) 
    - Maintainer: Jahid (email: islam034@umn.edu)
    - https://github.com/xahidbuffon/tf-rgb-lab
"""
import argparse
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
# local library
import rgb_lab_formulation as Conv_img


def test_tf_rgb_lab(img, tf_v1=True):
    # raw tensor
    raw_input = tf.image.convert_image_dtype(img, dtype=tf.float32)
    raw_input.set_shape([None, None, 3])

    # convert to lab-space image {L, a, b}
    lab = Conv_img.rgb_to_lab(raw_input)
    L_chan, a_chan, b_chan = Conv_img.preprocess_lab(lab)
    lab = Conv_img.deprocess_lab(L_chan, a_chan, b_chan)

    # get back the RGB image (tensor)
    true_image = Conv_img.lab_to_rgb(lab)
    true_image = tf.image.convert_image_dtype(true_image, dtype=tf.uint8, saturate=True)

    # get image array from tensor
    if tf_v1: # for tf v1.x
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            image = true_image.eval()
    else: # for tf v2.0
        tf.compat.v1.disable_eager_execution()
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            image = true_image.numpy()
    # save/show image
    plt.imshow(image)
    plt.imsave('output.jpg', image)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_path', dest='im_path', type=str, default='data/umn.jpg')
    args = parser.parse_args()
    img = plt.imread(args.im_path)
    test_tf_rgb_lab(img, tf_v1=True)

