### Modules for color-space conversion

- RGB to LAB and LAB to RGB
- Las tested on: Python 2.7, SciPy 0.18.1, TensorFlow 1.4.0


### Usage
Use the [test_.py](test_.py) test_.py for testing. 
The color-space conversion libraries are in [rgb_lab_formulation.py](rgb_lab_formulation.py). 


### Sample output

| Original Input image | RGB-LAB-RGB | 
|:--------------------|:--------------------
| ![det-7](/data/umn.jpg)) |  ![det-2](/data/converted_umn.jpg) |


### Acknowledgements
- https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py 
- https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c 
- https://github.com/cameronfabbri/Colorizing-Images-Using-Adversarial-Networks 
