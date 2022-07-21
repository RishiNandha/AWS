from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Dropout, SpatialDropout2D, AveragePooling2D
import numpy as np
import pickle

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"


print('\n\nLoaded Prepared Data Successfully')

layerz = [Conv2D(18, kernel_size=(5,5), input_shape=(160,160,3),activation='relu'),
		MaxPooling2D(pool_size=(2,2)), 
		Conv2D(36, kernel_size=(3,3),activation='relu'),  
		Conv2D(50, kernel_size=(3,3),activation='relu'),  
		MaxPooling2D(pool_size=(2,2)), 
		Conv2D(50, kernel_size=(3,3),activation='relu'),
		Conv2D(50, kernel_size=(3,3),activation='relu'),
		MaxPooling2D(pool_size=(2,2)), 
		Conv2D(50, kernel_size=(3,3),activation='relu'), 
		Conv2D(50, kernel_size=(3,3),activation='relu'),
		Conv2D(50, kernel_size=(3,3),activation='relu'), 
		Conv2D(50, kernel_size=(3,3),activation='relu'),
		Conv2D(36, kernel_size=(3,3),activation='relu'), 
		Conv2D(36, kernel_size=(3,3),activation='relu'),  
		MaxPooling2D(pool_size=(2,2)),  
		Flatten(), 
		Dense(72, activation='relu'),
		Dense(6,activation='sigmoid')]

model=keras.Sequential(layerz)
[]
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.000005), metrics=['accuracy'])

print('\n \nTraining: \n \n')

checkpoint = keras.callbacks.ModelCheckpoint('ckpt.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

from keras.preprocessing.image import ImageDataGenerator
# Generator Object with transformation settings
train_datagen = ImageDataGenerator(
        rescale=1./255,     shear_range=0.2,
        zoom_range=0.2,     horizontal_flip=True,
        vertical_flip=True, width_shift_range=0.2,
        height_shift_range=0.2)
# Vectorize Images in Training Directory
train_generator = train_datagen.flow_from_directory(
        'train',           target_size=(160, 160),
        batch_size=32,      class_mode='categorical')

#Similarly generate validation set too called validation_generator

# Train Model
model.fit_generator(
        train_generator, steps_per_epoch=2000,
        epochs=20)

model.save("AWS")

print('\n\nSaving Model Successful')