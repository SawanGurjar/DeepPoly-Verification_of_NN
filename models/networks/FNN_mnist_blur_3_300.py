import tensorflow as tf
import pandas as pd
import numpy as np

df=pd.read_csv('networks/dataset/mnist_blur_floatshift60000.csv',skiprows=0)		# loads MNIST blured image dataset.
blurdata = np.array(df)

def FFNN(blurdata):
	x_train = np.reshape(blurdata[:50000, 1:785], (50000, 28, 28))
	y_train = np.reshape(blurdata[:50000, :1], (50000,1))

	x_test = np.reshape(blurdata[50000:, 1:785], (10000, 28, 28))
	y_test = np.reshape(blurdata[50000:, :1], (10000,1))	
	x_train, x_test = x_train / 255.0, x_test / 255.0			# Normalizes the blured images.

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

	model.fit(x_train, y_train, epochs=15)						# train the model iterating entire dataset for 5 epochs.
	model.evaluate(x_test, y_test)								# evaluate the model on the test data.
	return model

net = FFNN(blurdata)
tf.saved_model.save(net, "FNN_mnist_blur_3_300")
# FFNN_mnist_3_300_model.pb is tensorflow model includes graph definitions and metadata of the model.
# variables are files that hold the serialized variables of the graphs.