from packaging import version
import tensorflow as tf

def testTensorFlowInstallation():
   hello = tf.constant('Hello, TensorFlow!')
   print('TensorFlow version: ', tf.__version__)

   if tf.test.is_built_with_cuda():
      print('TensorFlow was built with CUDA (GPU) support.')
   else:
      print('TensorFlow was built without CUDA (GPU) support.')

   if version.parse(tf.__version__) >= version.parse("2.0.0"):
      print('tf.Session() not supported')
      print(hello)
   else:
      sess = tf.Session()
      print(sess.run(hello))

