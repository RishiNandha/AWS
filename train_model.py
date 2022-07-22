from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Dropout, SpatialDropout2D, AveragePooling2D
import numpy as np
import pickle
x_val=pickle.load(open('x_val.bin','rb'))
y_val=pickle.load(open('y_val.bin','rb'))

layerz = [Conv2D(18, kernel_size=(5,5), input_shape=(300,300,3),activation='relu'), 
		Conv2D(36, kernel_size=(3,3),activation='relu'),  
		Conv2D(50, kernel_size=(3,3),activation='relu'),  
		MaxPooling2D(pool_size=(2,2)), 
		Conv2D(50, kernel_size=(3,3),activation='relu'),
		Conv2D(50, kernel_size=(3,3),activation='relu'),
		Conv2D(50, kernel_size=(3,3),activation='relu'),
		MaxPooling2D(pool_size=(2,2)),  
		Conv2D(50, kernel_size=(3,3),activation='relu'),
		Conv2D(50, kernel_size=(3,3),activation='relu'), 
		Conv2D(50, kernel_size=(3,3),activation='relu'),
		MaxPooling2D(pool_size=(2,2)),
		Conv2D(50, kernel_size=(3,3),activation='relu'), 
		Conv2D(50, kernel_size=(3,3),activation='relu'),
		Conv2D(50, kernel_size=(3,3),activation='relu'),    
		MaxPooling2D(pool_size=(2,2)),  
		Flatten(), 
		Dense(100, activation='relu'),
		Dense(6,activation='sigmoid')]

model=keras.Sequential(layerz)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.00005), metrics=['accuracy'])

print('\n \nTraining: \n \n')
	
checkpoint = keras.callbacks.ModelCheckpoint('ckpt.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

#model.load_weights('ckpt.hdf5')

from keras.preprocessing.image import ImageDataGenerator
# Generator Object with transformation settings
train_datagen = ImageDataGenerator(
        rescale=1./255,     shear_range=0.1,
        zoom_range=0.2,     horizontal_flip=True,
        vertical_flip=True, width_shift_range=0.2,
        height_shift_range=0.2)
# Vectorize Images in Training Directory
batch=12

train_generator = train_datagen.flow_from_directory(
        'train',           target_size=(300, 300),
        batch_size=batch,      class_mode='categorical')

#Similarly generate validation set too called validation_generator

# Train Model
model.fit(
        train_generator, steps_per_epoch=1995//batch,
        epochs=50, validation_data=(x_val,y_val), callbacks=[checkpoint])

model.save("AWS")

print('\n\nSaving Model Successful')