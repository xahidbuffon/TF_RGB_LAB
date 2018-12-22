#!/usr/bin/env python
"""
Maintainer: Jahid (email: islam034@umn.edu)
Interactive Robotics and Vision Lab
http://irvlab.cs.umn.edu/
Any part of this repo can be used for academic and educational purposes only
"""


from scipy import misc
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse

# local library
import rgb_lab_formulation as Conv_img


if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--im', required=False, dest='im', type=str, default='data/umn.jpg', help='image path')
      args = parser.parse_args()

      img = misc.imread(args.im)
      # raw tensor
      raw_input = tf.image.convert_image_dtype(img, dtype=tf.float32)
      raw_input.set_shape([None, None, 3])

      # convert to lab-space image {L, a, b}
      lab = Conv_img.rgb_to_lab(raw_input)
      L_chan, a_chan, b_chan = Conv_img.preprocess_lab(lab)
      lab = Conv_img.deprocess_lab(L_chan, a_chan, b_chan)

      # get back the RGB image
      true_image = Conv_img.lab_to_rgb(lab)
      true_image = tf.image.convert_image_dtype(true_image, dtype=tf.uint8, saturate=True)

      init_op = tf.global_variables_initializer()
      with tf.Session() as sess:
	  sess.run(init_op)
          # here is the image Tensor :)
	  image = true_image.eval()  
	  conv_img = Image.fromarray(image, 'RGB')
	  
	  #conv_img.save('converted_test.jpg')
	  conv_img.save('test.jpg')
	  #conv_img.show()
