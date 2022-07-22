import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

os.chdir('predb')

y_val = np.array([]).reshape((0,6))
x_val = np.array([]).reshape((0,300,300,3))

def encode_class(x):
	l = [0 for i in range(6)]
	l[x]=1
	return np.array(l).reshape((1,6))

def load_data(name):
	global x_val
	global y_val

	x=5

	image = plt.imread(name)
	image = cv2.resize(image,(300,300))

	x_val = np.append(x_val, image.reshape((1,300,300,3)),axis=0)

	if 'cardboard' in name:
		x=0
	elif 'glass' in name:
		x=1
	elif 'metal' in name:
		x=2
	elif 'paper' in name:
		x=3
	elif 'plastic' in name:
		x=4

	y_val = np.append(y_val, encode_class(x),axis=0)

for i in os.listdir()[1::2]:
	load_data(i)

print('\n',x_val.shape, '\t', y_val.shape)
os.chdir("..")

import pickle

with open("x_val.bin", 'wb') as f:
	pickle.dump(x_val,f)

with open("y_val.bin", 'wb') as f:
	pickle.dump(y_val,f)

print('\n\nPrepared')

