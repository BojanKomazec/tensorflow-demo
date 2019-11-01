from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt

def mnistDemo():
	# Load MNIST dataset, a collection of grayscale 28x28px images of handwritten digits.
	# The MNIST database contains 60,000 training images and 10,000 testing images.
	mnist = tf.keras.datasets.mnist

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	#
	# Dataset exploration
	#

	print("x_train.shape:\n", x_train.shape) # (60000, 28, 28)
	print("x_train:\n", x_train)
	print("x_train[0]:\n", x_train[0])

	print("y_train.shape:\n", y_train.shape) # (60000,)
	print("y_train:\n", y_train)             # [5 0 4 ... 5 6 8]
	print("y_train[0]:\n", y_train[0])       # 5

	print("x_test.shape:\n", x_test.shape)   # (10000, 28, 28)
	print("x_test:\n", x_train)
	print("y_test.shape:\n", y_test.shape)   # (10000,)
	print("y_test:\n", y_train)

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