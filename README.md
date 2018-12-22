### Modules for color-space conversion

- RGB to LAB and LAB to RGB
- Las tested on: Python 2.7, SciPy 0.18.1, TensorFlow 1.4.0


### Usage
- The test script: [test.py](test.py) 
- Color-space conversion libraries: [rgb_lab_formulation.py](rgb_lab_formulation.py)


### Sample output

| Original image  | RGB-to-LAB-to-RGB | 
|:--------------------|:----------------
| ![det-86](/data/test_rgb.jpg) |   ![det-106](/data/converted_test.jpg) | 
| ![det-86](/data/umn.jpg) |   ![det-106](/data/converted_umn.jpg) | 


### Acknowledgements
- https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py 
- https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c 
- https://github.com/cameronfabbri/Colorizing-Images-Using-Adversarial-Networks 
