import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

os.chdir('predb')

y_test = np.array([]).reshape((0,6))
x_test = np.array([]).reshape((0,256,256,3))

def encode_class(x):
	l = [0 for i in range(6)]
	l[x]=1
	return np.array(l).reshape((1,6))

def load_data(name):
	global x_test
	global y_test

	x=5

	image = plt.imread(name)
	image = cv2.resize(image,(256,256))

	x_test = np.append(x_test, image.reshape((1,256,256,3)),axis=0)

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

	y_test = np.append(y_test, encode_class(x),axis=0)

for i in os.listdir()[::2]:
	load_data(i)

print('\n',x_test.shape, '\t', y_test.shape)
os.chdir("..")
import pickle

with open("x_test.bin", 'wb') as f:
	pickle.dump(x_test,f)

with open("y_test.bin", 'wb') as f:
	pickle.dump(y_test,f)

print('\n\nPrepared')

