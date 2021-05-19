import tensorflow as tf
mnist = tf.keras.datasets.mnist					# MNIST handwritten digits dataset.

def FFNN():
	(x_train, y_train),(x_test, y_test) = mnist.load_data()		# loads the MNIST dataset.
	x_train, x_test = x_train / 255.0, x_test / 255.0			# Normalizes the images.

	# model cloning
	# class Sequential groups a linear stack of layers into a tf.keras.Model.
	model = tf.keras.models.Sequential([
				tf.keras.layers.Flatten(input_shape=(28, 28)),		# Flattens the input. Does not affect the batch size.
				tf.keras.layers.Dense(300, activation='relu'),		# Just your regular densely-connected NN layer.
				tf.keras.layers.Dense(300, activation='relu'),
				tf.keras.layers.Dense(300, activation='relu'),
				tf.keras.layers.Dense(10, activation='softmax')
				])

	model.compile(optimizer='adam',
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy'])

	model.fit(x_train, y_train, epochs=2)						# train the model iterating entire dataset for 5 epochs.
	model.evaluate(x_test, y_test)								# evaluate the model on the test data.
	return model

net = FFNN()
tf.saved_model.save(net, "net3")
# FFNN_mnist_3_300_model.pb is tensorflow model includes graph definitions and metadata of the model.
# variables are files that hold the serialized variables of the graphs.