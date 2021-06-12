import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf 
from tensorflow import keras


import dataLoader
import Generator
import Discriminator

generator = Generator.Generator()
discriminator = Discriminator.Discriminator()
train_ds = dataLoader.train_dataset


#define loss and optimizers
loss = keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


#Save checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


                            
from tqdm import trange, tqdm
EPOCHS = 50
BATCH_SIZE = 32
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)

seed = tf.random.normal([num_examples_to_generate, noise_dim])

for epoch in trange(EPOCHS):
    for input_image in train_ds:

      #print(input_image.shape)

      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          noise = tf.random.normal([BATCH_SIZE, noise_dim])

          generated_image = generator(noise, training = True)
          
          D_real = discriminator(input_image, training = True)
          D_fake = discriminator(generated_image, training = True)

          D_loss_real = loss(D_real,  tf.ones_like(D_real))
          D_loss_fake = loss(D_fake, tf.zeros_like(D_fake))
          D_total_loss = D_loss_fake + D_loss_real

          G_loss = loss(D_fake, tf.ones_like(D_fake) )
      

      generator_gradients = gen_tape.gradient(G_loss,
                                          generator.trainable_variables)
      discriminator_gradients = disc_tape.gradient(D_total_loss,
                                              discriminator.trainable_variables)

      generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
      discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables)) 

    if (epoch + 1) % 400 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

checkpoint.save(file_prefix=checkpoint_prefix)
