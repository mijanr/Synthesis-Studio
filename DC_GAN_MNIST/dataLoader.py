# import necessary libraries
import tensorflow as tf
from tensorflow import keras

#load the dataSet from the keras dataset directory
(train_images, target_images), (test_images, test_target) = keras.datasets.mnist.load_data()

#Shapes
'''print("train image shape: ", train_images.shape)
print("target image shape: ", target_images.shape)
print("test image shape: ", test_images.shape)
print("test target shape: ", test_target.shape)
'''
# reshape the train_image to have a fourth dimension and then normalize them between [-1,1]
shape = train_images.shape 
train_images = train_images.reshape(shape[0], shape[1], shape[2], 1)
train_images = (train_images-127.5)/127.5

#create a dataLoader, make batches of data and shuffle them
BUFFER_SIZE = 60000
BATCH_SIZE = 32
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

