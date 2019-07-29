import tensorflow as tf

def testTensorFlowInstallation():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))
