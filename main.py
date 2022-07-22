import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from matplotlib.pyplot import imread
from cv2 import resize
import numpy as np

model = keras.models.load_model("AWS")
l=["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Mixed"]

flag = 1
while flag==1:
	img = imread(input("\n\nEnter Full Address of Image to predict: "))
	img = resize(img,(256,256))
	img = img.reshape((1,256,256,3))
	print(' ')
	img = model.predict(img,verbose=0)
	img=img.reshape((6,))

	print(l[int(np.where(img == np.amax(img))[0][0])])
	print((np.amax(img)*100)//1, "%")

	flag=int(input("\n\nContinue? (1/0) : "))