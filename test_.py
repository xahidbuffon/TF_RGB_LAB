from scipy import misc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

import rgb_lab_formulation as Conv_img


if __name__ == '__main__':
      
      #img = misc.imread('test_rgb.jpg')
      img = misc.imread('umn.jpg')

      raw_input = tf.image.convert_image_dtype(img, dtype=tf.float32)
      raw_input.set_shape([None, None, 3])

      lab = Conv_img.rgb_to_lab(raw_input)
      L_chan, a_chan, b_chan = Conv_img.preprocess_lab(lab)

      lab = Conv_img.deprocess_lab(L_chan, a_chan, b_chan)
      true_image = Conv_img.lab_to_rgb(lab)
      true_image = tf.image.convert_image_dtype(true_image, dtype=tf.uint8, saturate=True)

      init_op = tf.global_variables_initializer()
      with tf.Session() as sess:
	  sess.run(init_op)
	  image = true_image.eval() #here is your image Tensor :) 

	  conv_img = Image.fromarray(image, 'RGB')
	  
	  #conv_img.save('converted_test.jpg')
	  conv_img.save('converted_umn.jpg')
	  #conv_img.show()
