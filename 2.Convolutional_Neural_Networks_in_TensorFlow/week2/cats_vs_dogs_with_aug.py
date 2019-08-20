# %%
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from supportfiles.create_and_fill_dirs import fill_dirs


# %%
fill_dirs(os.path.join(Path.home(), 'Downloads/kagglecatsanddogs_3367a/PetImages'), 'cats_vs_dogs', 0.9)
print(len(os.listdir(os.path.join(os.getcwd(), 'supportfiles/cats_vs_dogs/training/Cat'))))
print(len(os.listdir(os.path.join(os.getcwd(), 'supportfiles/cats_vs_dogs/training/Dog'))))
print(len(os.listdir(os.path.join(os.getcwd(), 'supportfiles/cats_vs_dogs/testing/Cat'))))
print(len(os.listdir(os.path.join(os.getcwd(), 'supportfiles/cats_vs_dogs/testing/Dog'))))
# %%
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
#%%
model.compile(loss='binary_crossentropy',
              loptimizer=RMSprop(lr=1e-4),
              metrics=['acc'])