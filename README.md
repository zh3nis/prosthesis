## Depth Image Based Intent Recognition for Lower Limb Prostheses with C3D Model
Code for the [C3D model](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.html) applied to the data from the paper [A Feasibility Study of Depth Image Based Intent Recognition for
Lower Limb Prostheses](http://ieeexplore.ieee.org/abstract/document/7591863/) (EMBC 2016)

### Requirements
Code is written in Python 2.7 and requires TensorFlow 1.4+. It also requires the following Python modules: `numpy`, `imageio`, `Pillow`, `opencv-python`, . You can install them via:
```
pip2 install numpy imageio Pillow opencv-python
```

### Data
Data should be put into the `database/` directory, split into trial sets `set1`, `set2`, etc. Each folder `seti` should contain (... description of folder ...). 

### Configuration
Hyperparameters are specified in the `Config` class at the beginning of `prosthesis.py` script. Some of them are discussed in the [shared Google document](https://docs.google.com/document/d/1i5ORk2fcvnN_9pVVEOOmZUlIZ6-QXRFM6lwp9zh-2g8/edit?usp=sharing).