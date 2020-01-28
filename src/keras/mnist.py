from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
# import tkinter
import matplotlib
# matplotlib.use('Agg')
# matplotlib.use('Gtk3Agg')
import matplotlib.pyplot as plt
import pydoc
import inspect

def mnistDemo():
	print("TensofFlow version: {}. Install location: {}".format(tf.__version__, tf.__file__))

	# Load MNIST dataset, a collection of grayscale 28x28px images of handwritten digits.
	# The MNIST database contains 60,000 training images and 10,000 testing images.
	mnist = tf.keras.datasets.mnist
	print("type(mnist): {}".format(type(mnist)))
	print("dir(mnist): {}".format(dir(mnist)))
	# print("help(mnist): {}".format(help(mnist))) # opens interactive Python
	print("pydoc.render_doc(mnist): {}".format(pydoc.render_doc(mnist))) # this gives the same outputu as help()
	print("inspect.signature(mnist.load_data): {}".format(inspect.signature(mnist.load_data)))
	print("pydoc.render_doc(mnist.load_data): {}".format(pydoc.render_doc(mnist.load_data)))

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	#
	# Dataset exploration
	#

	print("x_train.shape:\n", x_train.shape) # (60000, 28, 28)
	# print("x_train:\n", x_train)
	print("x_train[0]:\n", x_train[0])
	print("matplotlib version: {}. Install location: {}".format(matplotlib.__version__, matplotlib.__file__))
	print("matplotlib.get_backend(): {}".format(matplotlib.get_backend()))
	print("matplotlib.matplotlib_fname(): {}".format(matplotlib.matplotlib_fname()))
	# plt.imshow(x_train[0])
	plt.plot([1, 2, 3])
	plt.show()
	exit(0) # test

	print("y_train.shape:\n", y_train.shape) # (60000,)
	# print("y_train:\n", y_train)             # [5 0 4 ... 5 6 8]
	print("y_train[0]:\n", y_train[0])       # 5

	print("x_test.shape:\n", x_test.shape)   # (10000, 28, 28)
	# print("x_test:\n", x_test)
	print("y_test.shape:\n", y_test.shape)   # (10000,)
	# print("y_test:\n", y_test)


	#
	# Dataset visualization
	#

	first_array=x_train[0]
	#Not sure you even have to do that if you just want to visualize it
	#first_array=255*first_array
	#first_array=first_array.astype("uint8")
	plt.imshow(first_array)
	#Actually displaying the plot if you are not in interactive mode
	plt.show()
	#Saving plot
	plt.savefig("fig.png")

	#
	# Dataset validation
	#

	assert(x_train.shape[0] == y_train.shape[0]), "The number of images does not match the number of labels in the training set."
	assert(x_test.shape[0] == y_test.shape[0]), "The number of images does not match the number of labels in the test set."
	assert(x_train.shape[1:] == (28, 28)), "The dimension of the images is not 28x28 pixels."
	assert(x_test.shape[1:] == (28, 28)), "The dimension of the images is not 28x28 pixels."

	# prepare the MNIST dataset. Convert the samples from integers to floating-point numbers:
	x_train, x_test = x_train / 255.0, x_test / 255.0

	# Build the tf.keras.Sequential model by stacking layers.
	# Dense is a linear operation in which every input is connected to every output by a weight.

	model = tf.keras.models.Sequential([
		tf.keras.layers.Flatten(input_shape=(28, 28)),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dropout(0.2),
		tf.keras.layers.Dense(10, activation='softmax')
	])

	# Choose an optimizer and loss function for training:
	model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'])

	# Train and evaluate the model:

	model.fit(x_train, y_train, epochs=5)

	model.evaluate(x_test,  y_test, verbose=2)