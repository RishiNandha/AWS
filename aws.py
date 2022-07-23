import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from matplotlib.pyplot import imread, show, imshow
from cv2 import resize
import numpy as np

model = keras.models.load_model("AWS")
l=["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Mixed"]

flag = 1
os.chdir('predb')
dirlist=os.listdir()
print(' ')
while flag==1:

	# Open Image and Directory
	imgdir=dirlist[int(input("\nEnter Test Image Index (Enter an Integer lesser than "+str(len(dirlist))+") : "))]
	img = imread(imgdir)
	
	# Display Image
	imshow(img)
	show()
	
	# Run Image Through CNN
	img = resize(img,(256,256))
	img = img.reshape((1,256,256,3))
	img = model.predict(img,verbose=0)
	img=img.reshape((6,))

	# Correct Class
	print("\nCorrect Class :", imgdir)

	# Prediction by CNN
	if np.amax(img)<(2.0/3):
		print("\nPredicted Class : Mixed")
		print("Confidence was too low, hence it's precautiously categorized under Mixed")
	else:
		print("\nPredicted Class :",l[int(np.where(img == np.amax(img))[0][0])])
		print("Confidence :",(np.amax(img)*100)//1, "%")

	flag=int(input("\n\nContinue? (1/0) : "))

	print("\n----------------------------------------------------------------------------------------------------------------------")