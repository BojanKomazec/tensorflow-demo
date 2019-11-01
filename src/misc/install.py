import tensorflow as tf

from packaging import version

def testTensorFlowInstallation():
   hello = tf.constant('Hello, TensorFlow!')
   print('TensorFlow version: ', tf.__version__)

   if tf.test.is_built_with_cuda():
      print('TensorFlow was built with CUDA (GPU) support.')
   else:
      print('TensorFlow was built without CUDA (GPU) support.')

   if version.parse(tf.__version__) >= version.parse("2.0.0"):
      print('INFO: tf.Session() not supported')
      print(hello)
   else:
      sess = tf.Session()
      print(sess.run(hello))

