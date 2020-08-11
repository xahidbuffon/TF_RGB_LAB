### RGB-to-LAB and LAB-to-RGB color-space conversion
- tested on TensorFlow 1.x (Python 2.7) and TensorFlow 2.0 (Python 3.6) 

| Original image  | RGB-to-LAB-to-RGB | 
|:--------------------|:----------------
| ![det-86](/data/dance.jpg) |   ![det-106](/data/converted_dance.jpg) | 
| ![det-86](/data/umn.jpg) |   ![det-106](/data/converted_umn.jpg) | 


### Usage
- Test script: [test.py](test.py) (use the tf_v1 = True/False for tf 1.x/2.0)
- Color-space conversion libraries: [rgb_lab_formulation.py](rgb_lab_formulation.py)

#### Acknowledgements for some functionalities
- https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py 
- https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c 
- https://github.com/cameronfabbri/Colorizing-Images-Using-Adversarial-Networks 
