import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

y_train = np.array([]).reshape((0,6))
x_train = np.array([]).reshape((0,300,300,3))

def encode_class(x):
	l = [0 for i in range(6)]
	l[x]=1
	return np.array(l).reshape((1,6))

def append_data(directory, classification):
	
	os.chdir(directory)
	
	y_temp = np.array([]).reshape((0,6))
	x_temp = np.array([]).reshape((0,300,300,3))
	
	for i in os.listdir():
		y_temp2 = np.array([]).reshape((0,6))
		x_temp2 = np.array([]).reshape((0,300,300,3))
		image = plt.imread(i)
		y=encode_class(classification)
		
		image = cv2.resize(image,(300,300))
		x_temp2 = np.append(x_temp2, image.reshape((1,300,300,3)), axis=0)
		y_temp2 = np.append(y_temp2, y,axis=0)
		
		image = cv2.flip(image,0)
		x_temp2 = np.append(x_temp2, image.reshape((1,300,300,3)), axis=0)
		y_temp2 = np.append(y_temp2, y,axis=0)

		x_temp = np.append(x_temp, x_temp2, axis=0)
		y_temp = np.append(y_temp, y_temp2,axis=0)
	
	global x_train
	global y_train

	x_train = np.append(x_train, x_temp, axis=0)
	y_train = np.append(y_train, y_temp, axis=0)

	os.chdir("../..")
	
	print("\nData from", directory, "loaded.")
	print(x_train.shape,'\t', y_train.shape)

print("\n \nPreparing Data:\n")

append_data("train/metal",2)
append_data("train/cardboard",0)
append_data("train/glass",1)
append_data("train/paper",3)
append_data("train/plastic",4)
append_data("train/trash",5)


import pickle

with open("x_train.bin", 'wb') as f:
	pickle.dump(x_train,f)

with open("y_train.bin", 'wb') as f:
	pickle.dump(y_train,f)

print("\nPreparing data successful")