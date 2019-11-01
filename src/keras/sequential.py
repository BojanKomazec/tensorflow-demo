import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers

def sequentialNNDemo():
	model = tf.keras.Sequential()
	model.add(layers.Dense(64, activation='relu', input_shape=(32,)))
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(10, activation='softmax'))

	# Configure a model for categorical classification.
	model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
		loss=tf.keras.losses.CategoricalCrossentropy(),
		metrics=[tf.keras.metrics.CategoricalAccuracy()])

	import numpy as np
	data = np.random.random((1000, 32))
	labels = np.random.random((1000, 10))
	model.fit(data, labels, epochs=10, batch_size=32)


	# With a Dataset
	dataset = tf.data.Dataset.from_tensor_slices((data, labels))
	dataset = dataset.batch(32)

	model.evaluate(dataset)

	result = model.predict(data, batch_size=32)
	print(result.shape)