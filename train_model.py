from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Dropout, SpatialDropout2D, AveragePooling2D
import numpy as np
import pickle

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

x_train=pickle.load(open('x_train.bin','rb'))
y_train=pickle.load(open('y_train.bin','rb'))

x_val=pickle.load(open('x_val.bin','rb'))
y_val=pickle.load(open('y_val.bin','rb'))

print('\n\nLoaded Prepared Data Successfully')

layerz = [Conv2D(18, kernel_size=(5,5), input_shape=(192,256,3),activation='relu'),
		AveragePooling2D(pool_size=(2,2)),
		Conv2D(36, kernel_size=(3,3),activation='relu'),  
		MaxPooling2D(pool_size=(2,2)), 
		Conv2D(50, kernel_size=(3,3),activation='relu'),  
		MaxPooling2D(pool_size=(2,2)), 
		Conv2D(50, kernel_size=(3,3),activation='relu'),
		Conv2D(50, kernel_size=(3,3),activation='relu'),
		MaxPooling2D(pool_size(3,3)), 
		Conv2D(50, kernel_size=(3,3),activation='relu'), 
		Conv2D(50, kernel_size=(3,3),activation='relu'),  
		MaxPooling2D(pool_size=(2,2)), 
		Flatten(), 
		Dense(200, activation='relu'),
		Dense(6,activation='sigmoid')]

model=keras.Sequential(layerz)

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.00003), metrics=['accuracy'])

print('\n \nTraining: \n \n')

checkpoint = keras.callbacks.ModelCheckpoint('ckpt.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

# model.load_weights('ckpt.hdf5')
# USE ONLY IF CHECKPOINT EXISTS AND IS FROM THE SAME MODEL

model.fit(x_train, y_train, epochs=25, batch_size = 18, shuffle=True, callbacks=[checkpoint, stop], validation_data=(x_val, y_val))

model.load_weights('ckpt.hdf5')

model.save("AWS")

print('\n\nSaving Model Successful')