batch_size = 128
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
path1=r"E:\6th sem\tarp deep\dataset1\test"
classes_train=os.listdir(path1)

train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
        'dataset1/train', 
        target_size=(200, 200),  
        batch_size=batch_size,
        classes = classes_train,
        class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1/255)
val_generator = val_datagen.flow_from_directory(
        'dataset1/val', 
        target_size=(200, 200),  
        batch_size=batch_size,
        classes = classes_train,
        class_mode='categorical')

import tensorflow as tf
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),    
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(128, activation='relu'),
    
    tf.keras.layers.Dense(35, activation='softmax')
])
model.summary()
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.0005),
              metrics=['acc'])

total_sample=train_generator.n
n_epochs = 30

checkpoint = keras.callbacks.ModelCheckpoint('abc.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

history = model.fit(
        train_generator, 
        steps_per_epoch=int(total_sample/batch_size),  
        epochs=n_epochs,
        validation_data = val_generator,
        callbacks = [checkpoint],
        verbose=1)


from tensorflow.keras.models import load_model
model = load_model('abc.h5')
# summarize model.
model.summary()

import numpy as np
from keras.preprocessing import image
test_image = image.load_img(r'E:\6th sem\tarp deep\dataset1\test\Corn___Common_rust\image (201).JPG', target_size = (200, 200))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
model.evaluate_generator(val_generator)