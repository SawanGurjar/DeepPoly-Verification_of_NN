import pandas as pd
import numpy as np
import matplotlib
import tensorflow as tf
import keras

from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

# Number of blured images you want to keep in your test dataset.
############## set according to your convenience...##############
num_test_images = 99

# Read the data from mnist dataset to train the network
df=pd.read_csv('dataset/mnist_test.csv',skiprows=0)

# Train and reshape the dataset
x_train=np.array(df)
y_train=np.reshape(x_train[:num_test_images, :1], (num_test_images,1))
x_train = np.reshape(x_train[:num_test_images, 1:785], (num_test_images, 28, 28))



# Elastic deformation of image
def elastic_transform(image, alpha_range, sigma, random_state=None):
	"""
	# Arguments
		image: Numpy array with shape (height, width, channels). 
		alpha_range: Float for fixed value or [lower, upper] for random value from uniform distribution.
			Controls intensity of deformation.
		sigma: Float, sigma of gaussian filter that smooths the displacement fields.
		random_state: `numpy.random.RandomState` object for generating displacement fields.
	"""
	if random_state is None:
		random_state = np.random.RandomState(None)
		
	if np.isscalar(alpha_range):
		alpha = alpha_range
	else:
		alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

	shape = image.shape
	dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
	dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

	x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
	indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1)), np.reshape(z, (-1, 1))

	return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)



# Function to plot MNIST digits.
def plot_digits(examples, title=None, size_mult=1):
	"""
	# Arguments
		examples: Numpy array with shape (num_examples, height, width, num_iterations).
		title: Plot title string.
		size_mult: Multiply figsize by `size_mult`.
	"""
	num_iterations = examples.shape[-1]
	num_examples = examples.shape[0]    
	
	plt.rcParams['figure.figsize'] = (num_examples * size_mult, num_iterations * size_mult)
	plt.rcParams['image.interpolation'] = 'nearest'
	plt.rcParams['image.cmap'] = 'gray'
	
	for c in range(num_iterations):
		for i, ex in enumerate(examples):
			plt.subplot(num_iterations, num_examples, num_examples * c + i + 1)            
			plt.imshow(ex[:,:,c])  
			plt.axis('off')
			if c == 0 and i == 0 and title is not None:
				plt.text(
					x=0,
					y=-ex.shape[1] // 4 // size_mult,
					s=title,
					fontsize=13,
					horizontalalignment='left', 
					verticalalignment='bottom')

	plt.show()


	
# Plot output after elastic distortion and select Keras data augmentations.
def plot_augmented(examples, alpha_range=0, sigma=0, 
					width_shift_range=0, height_shift_range=0, zoom_range=0.0, 
					iterations=1, title=None, size_mult=1, shift_type=None):
	"""
	# Arguments
		examples: Numpy array with shape (num_examples, height, width, num_iterations).
		alpha_range, sigma: arguments for `elastic_transform()`.
		width_shift_range, height_shift_range, zoom_range: arguments for Keras `ImageDataGenerator()`.
		iterations: Int, number of times to randomly augment the examples.
		title: Plot title string.
		size_mult: Multiply figsize by `size_mult`.
	"""
	datagen = keras.preprocessing.image.ImageDataGenerator(
		width_shift_range=width_shift_range, 
		height_shift_range=height_shift_range, 
		zoom_range=zoom_range,  
		preprocessing_function=lambda x: elastic_transform(x, alpha_range=alpha_range, sigma=sigma)
		)
	x = [datagen.flow(examples, batch_size=num_test_images, shuffle=False).next() for i in range(iterations)]
	x = np.concatenate(x, axis=-1)

	# plots the blured digits
	plot_digits(x, title=title, size_mult=size_mult)

	all_images = x.copy()						# copy of blured image data(4D array)
	final_op = np.array([])						# 2D array to store flatten values(1, 784) of each image as an element.
	for i in range(num_test_images):
		flat_img = all_images[i].flatten()		# flatten each image data(28,28,1) into(1,784)
		if i==0:
			final_op = flat_img					# insert first image(1, 784) as 1D array into 2D array.
		else:
			final_op = np.vstack([final_op,flat_img])	# insert next image as 1D array

	# Add respective image label(image-class) as first element before each image
	final_op_labeled = np.hstack([y_train, final_op])

	# save the data as .csv to use as test data for blured image.
	np.savetxt('mnist_blur_{}shift.csv'.format(shift_type), final_op_labeled, delimiter=",")


b_examples = np.expand_dims(x_train[:num_test_images], -1)

# draws input image from dataset.
plot_digits(b_examples, title='Input Images', size_mult=2)

# Choose Function.
"""
	# Arguments
		To save Integer-Shift or Float-Shift images in .csv file, 
		consider respective commented function Plot_augmented() as the part of code(uncommented).
		Note that you can uncomment any one at a time i.e. comment the other function

		Also, You can change the values of parameters width_shift_range, height_shift_range
		to change the blurness of the images.
"""

# Integer Shift:
# plot_augmented(b_examples, width_shift_range=2, height_shift_range=2, title='Integer Shift', size_mult=2, shift_type='int')

# Float Shift:
plot_augmented(b_examples, width_shift_range=1., height_shift_range=1., title='Float Shift', size_mult=2, shift_type='float')