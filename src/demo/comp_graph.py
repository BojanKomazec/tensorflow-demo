import tensorflow as tf

def computationalGraphDemo():
    # Buiding computational graph
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0) # tf.float32 implicitly
    print(node1, node2)

    # Runnig computational graph
    sess = tf.Session()
    print(sess.run([node1, node2]))
