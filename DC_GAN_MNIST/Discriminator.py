from matplotlib.pyplot import disconnect
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers

#create Discriminator model
def Discriminator():
    model = keras.Sequential()
    model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding = 'same', input_shape=[28,28,1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding = 'same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

'''discriminator = Discriminator()
discriminator.summary()'''