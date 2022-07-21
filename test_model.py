from tensorflow import keras
import pickle

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

x_test = pickle.load(open('x_test.bin','rb'))
y_test = pickle.load(open('y_test.bin','rb'))

model = keras.models.load_model("AWS")

print('\n\nModel Retrieved')

print('\n\nTesting:\n\n')

model.evaluate(x_test, y_test)