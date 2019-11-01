# tensorflow-demo
Tensorflow demo project.

# Running in Docker container

To run this application in its Docker container it is first necessary to create its Docker image:
```
$ docker build -t tf-demo .
```
To launch td-demo container:
```
$ docker run -it --rm --name tf-demo tf-demo
```
# Running the application on native Ubuntu (dev machine)

The recommended way is to run it in virtual environment, just as it's recommended to install TensorFlow in it.
Once you clone this repository, install virtual environment for this project as it's described here: [Install TensorFlow with pip](https://www.tensorflow.org/install/pip).

To upgrade pip3 use:
```
$ pip3 install --upgrade pip
```

After installing virtual environment, there shall be `venv` directory in the root of the project.

Activate virtul environment wth:
```
$ source ./venv/bin/activate
```

Install all dependencies:
```
$ pip3 install -r requirements.txt
```

Unlike `venv` directory, `requirements.txt` is added to source control and it can be generated by doing the following:
```
$ pip3 freeze > requirements.txt
```

Run the application:
```
$ python3 src/tf-demo.py
```

Deactivate virtual environment:
```
$ deactivate
```


# Running in virtualenv

In order to run this script in virtualenv, activate virtualenv first:
```
../tensorflow/venv/tf-py3/bin$ source activate
```
Then run the script:
```
(tf-py3) ../tensorflow/venv/tf-py3/bin$ python ../../../demo-verify-tf-install/main.py 
```
Output shall be like this:
```
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
b'Hello, TensorFlow!'
```

# Installing Dependencies

## Tensorflow

```
import tensorflow as tf
```
requires having Tensorflow installed.

Many of the examples are written for TF version 1. To make them working on the box/venv with TF version 2 installed we need to do the following;

Replace
```
import tensorflow as tf
```
with:
```
import tensorflow.compat.v1 as tf
```
and execute:
```
tf.disable_v2_behavior()
```

```
$ pip3 install tensorflow
```


## Tensorflow Object Detection API

```
from utils import ...
```
requires having `<PATH_TO_TF>/TensorFlow/models/research/object_detection` added to Python module search path (among `research` and `research/slim`):
```
export PYTHONPATH=$PYTHONPATH:<PATH_TO_TF>/tensorflow/models/research:<PATH_TO_TF>/tensorflow/models/research/slim:<PATH_TO_TF>/TensorFlow/models/research/object_detection
```
If it is not added, we need to use:
```
from object_detection.utils import ...
```

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
https://github.com/tensorflow/models/blob/master/research/delf/INSTALL_INSTRUCTIONS.md
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html


## OpenCV

```
import cv2
```
requires having OpenCV installed:
```
$ sudo apt install python3-opencv
...
Setting up python3-numpy (1:1.13.3-2ubuntu1)
...
Setting up python3-opencv (3.2.0+dfsg-4ubuntu0.1)
...
```
If OpenCV is not installed the following error is issued:
```
unresolved import 'cv2'
```

## Other Packages

```
$ pip3 install opencv-python
$ pip3 install packaging
$ sudo apt install python3-opencv
$ sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
$ pip install --user Cython
$ pip install --user contextlib2
$ pip install --user jupyter
$ pip install --user matplotlib
$ git clone https://github.com/tensorflow/models.git
```

To avoid installing pacakges one by one a requirements.txt is provided.
It has been generated via:
```
$ pip3 install -r requirements.txt
```
Upon entering virtual environment:
```
$ source ./venv/bin/activate
```
to install all requirements execute:
```
pip3 install -r requirements.txt
```

### References

[Streaming Object Detection Video - Tensorflow Object Detection API Tutorial](https://pythonprogramming.net/video-tensorflow-object-detection-api-tutorial/)