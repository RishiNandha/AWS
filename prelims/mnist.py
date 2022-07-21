import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Dropout, SpatialDropout2D
import numpy as np
import pandas as pd 

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

def encode_digit(x):
	l=[0 for i in range(10)]
	l[x]=1
	return l 

y_train = np.array([encode_digit(i) for i in y_train])


y_test = np.array([encode_digit(i) for i in y_test])

layerz = [Conv2D(10, kernel_size=(5,5), input_shape=(28,28,1),activation='relu'), MaxPooling2D(pool_size=(4,4)), Conv2D(20, kernel_size=(3,3)), Flatten(), Dense(128, activation='relu'),Dropout(0.1), Dense(10,activation='sigmoid')]

model=keras.Sequential(layerz)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('\n \nTraining: \n \n')

model.fit(x_train, y_train, epochs=15, batch_size = 100)

print('Testing: \n \n')

model.evaluate(x_test, y_test)