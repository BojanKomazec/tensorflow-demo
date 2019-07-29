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
