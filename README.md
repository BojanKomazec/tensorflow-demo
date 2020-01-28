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
# Running the application in Virtual Environment on Ubuntu (dev box)

The recommended way is to run it in virtual environment, just as it's recommended to install TensorFlow in it.
Once you clone this repository, install virtual environment for this project as it's described in [Install TensorFlow with pip](https://www.tensorflow.org/install/pip):

```
$ sudo apt update
$ sudo apt install python3-dev python3-pip
$ sudo pip3 install -U virtualenv
$ virtualenv --system-site-packages -p python3 ./venv # creates `venv` directory in the root of the project
$ source ./venv/bin/activate
$ pip install --upgrade pip
$ pip list  # show packages installed within the virtual environment
```

---
Always make sure `pip` version 3 is used:
```
$ pip --version
pip 19.3.1 from /home/bojan/.local/lib/python3.6/site-packages/pip (python 3.6)
$ pip3 --version
pip 19.3.1 from /home/bojan/.local/lib/python3.6/site-packages/pip (python 3.6)
```
This means we can use `pip` interchangeably with `pip3`.

---

In the next step we need to install all dependencies - see [Installing Dependencies](#installing-dependencies).


Run the application:
```
$ python3 src/tf-demo.py
```

Deactivate virtual environment:
```
$ deactivate
```


# Installing Dependencies

Install all dependencies:
```
$ pip install -r requirements.txt
```

Unlike `venv` directory, `requirements.txt` is added to source control and it can be generated by doing the following:
```
$ pip freeze > requirements.txt
```

## Tensorflow

```
import tensorflow as tf
```
requires having Tensorflow installed:
```
$ pip install tensorflow
```

Many of the examples are written for TF version 1. To make them working on the box/venv with TF version 2 installed we need to do the following:

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

## Tensorflow Object Detection API

```
import object_detection
```
requires cloning models repository and adding `models/research/object_detection` to `PYTHONPATH`:

```
$ mkdir tensorflow
$ cd tensorflow
$ git clone https://github.com/tensorflow/models.git
$ export PYTHONPATH=$PYTHONPATH:`pwd`/models/research/object_detection
```

To rectify this error:
```
Traceback (most recent call last):
  File "src/tf-demo.py", line 17, in <module>
    main()
  File "src/tf-demo.py", line 14, in main
    webcamObjectDetectionDemo()
  File "/home/bojan/dev/github/tensorflow-demo/src/opencv/webcam_obj_detection.py", line 74, in webcamObjectDetectionDemo
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  File "/home/bojan/dev/github/tensorflow/models/research/object_detection/utils/label_map_util.py", line 138, in load_labelmap
    with tf.gfile.GFile(path, 'r') as fid:
AttributeError: module 'tensorflow' has no attribute 'gfile'
```
we need to edit line 138 of `research/object_detection/utils/label_map_util.py`:
```
-  with tf.gfile.GFile(path, 'r') as fid:
+  with tf.io.gfile.GFile(path, 'r') as fid:
```

```
from utils import ...
```
requires having `<PATH_TO_TF>/TensorFlow/models/research/object_detection` added to Python module search path (among `research` and `research/slim`):
```
export PYTHONPATH=$PYTHONPATH:<PATH_TO_TF>/tensorflow/models/research:<PATH_TO_TF>/tensorflow/models/research/slim
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
If OpenCV is not installed or its path is not available in the virtual environment, the following error is issued:
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

```
$ pip listPackage               Version
--------------------- -------------------
absl-py               0.8.1              
apturl                0.5.2              
asn1crypto            0.24.0             
astor                 0.8.0              
astroid               2.2.5              
attrs                 19.3.0             
backcall              0.1.0              
bleach                3.1.0              
Brlapi                0.6.6              
cachetools            3.1.1              
certifi               2018.1.18          
change-case           0.5.2              
chardet               3.0.4              
command-not-found     0.3                
contextlib2           0.6.0.post1        
cryptography          2.1.4              
cupshelpers           1.0                
cycler                0.10.0             
Cython                0.29.14            
decorator             4.4.1              
defer                 1.0.6              
defusedxml            0.6.0              
distro-info           0.18ubuntu0.18.04.1
entrypoints           0.3                
gast                  0.2.2              
google-auth           1.7.1              
google-auth-oauthlib  0.4.1              
google-pasta          0.1.8              
grpcio                1.25.0             
h5py                  2.10.0             
httplib2              0.9.2              
idna                  2.6                
importlib-metadata    1.1.0              
ipykernel             5.1.3              
ipython               7.10.1             
ipython-genutils      0.2.0              
ipywidgets            7.5.1              
iso8601               0.1.12             
isort                 4.3.17             
jedi                  0.15.1             
Jinja2                2.10.3             
jsonschema            3.2.0              
JSONSchema2DB         1.0.1              
jupyter               1.0.0              
jupyter-client        5.3.4              
jupyter-console       6.0.0              
jupyter-core          4.6.1              
Keras-Applications    1.0.8              
Keras-Preprocessing   1.1.0              
keyring               10.6.0             
keyrings.alt          3.0                
kiwisolver            1.1.0              
language-selector     0.1                
launchpadlib          1.10.6             
lazr.restfulclient    0.13.5             
lazr.uri              1.0.3              
lazy-object-proxy     1.3.1              
louis                 3.5.0              
macaroonbakery        1.1.3              
Mako                  1.0.7              
Markdown              3.1.1              
MarkupSafe            1.0                
matplotlib            3.1.2              
mccabe                0.6.1              
meld                  3.21.0             
mistune               0.8.4              
more-itertools        8.0.0              
nbconvert             5.6.1              
nbformat              4.4.0              
netifaces             0.10.4             
notebook              6.0.2              
numpy                 1.17.4             
oauth                 1.0.1              
oauthlib              3.1.0              
olefile               0.45.1             
opencv-python         4.1.2.30           
opt-einsum            3.1.0              
packaging             19.2               
pandocfilters         1.4.2              
parso                 0.5.1              
pbr                   5.4.3              
pexpect               4.2.1              
pickleshare           0.7.5              
Pillow                5.1.0              
pip                   19.3.1             
prometheus-client     0.7.1              
prompt-toolkit        2.0.10             
protobuf              3.11.0             
psycopg2              2.7.2              
ptyprocess            0.6.0              
pyasn1                0.4.8              
pyasn1-modules        0.2.7              
pycairo               1.18.1             
pycrypto              2.6.1              
pycups                1.9.73             
Pygments              2.5.2              
PyGObject             3.34.0             
pylint                2.3.1              
pymacaroons           0.13.0             
PyNaCl                1.1.2              
pyparsing             2.4.5              
pyRFC3339             1.0                
pyrsistent            0.15.6             
pysmbc                1.0.15.6           
python-apt            1.6.4              
python-dateutil       2.6.1              
python-debian         0.1.32             
pytz                  2018.3             
pyxdg                 0.25               
PyYAML                3.12               
pyzmq                 18.1.1             
qtconsole             4.6.0              
reportlab             3.4.0              
requests              2.22.0             
requests-oauthlib     1.3.0              
requests-unixsocket   0.1.5              
rsa                   4.0                
SecretStorage         2.3.1              
Send2Trash            1.5.0              
setuptools            42.0.2             
simplejson            3.13.2             
six                   1.12.0             
stevedore             1.31.0             
system-service        0.3                
systemd-python        234                
tensorboard           2.0.2              
tensorflow            2.0.0              
tensorflow-estimator  2.0.1              
termcolor             1.1.0              
terminado             0.8.3              
testpath              0.4.4              
tornado               6.0.3              
traitlets             4.3.3              
typed-ast             1.3.4              
ubuntu-drivers-common 0.0.0              
ufw                   0.36               
unattended-upgrades   0.1                
urllib3               1.22               
usb-creator           0.3.3              
virtualenv            16.7.7             
virtualenv-clone      0.5.3              
virtualenvwrapper     4.8.4              
wadllib               1.3.2              
wcwidth               0.1.7              
webencodings          0.5.1              
Werkzeug              0.16.0             
wheel                 0.33.6             
widgetsnbextension    3.5.1              
wrapt                 1.11.1             
xkit                  0.0.0              
zipp                  0.6.0              
zope.interface        4.3.2  
```


### References

[Streaming Object Detection Video - Tensorflow Object Detection API Tutorial](https://pythonprogramming.net/video-tensorflow-object-detection-api-tutorial/)
