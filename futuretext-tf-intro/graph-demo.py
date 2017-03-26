import tensorflow as tf

# Build graph nodes, starting from the inputs
a = tf.constant(5, name = "input_a")
b = tf.constant(3, name = "input_b")
c = tf.multiply(a, b, name = "mul_c")
d = tf.add(a, b, name = "add_d")
e = tf.add(c, d, name = "add_e")

# Create session
sess = tf.Session()

# Execute output node
output = sess.run(e)
print(output)
