import tensorflow as tf
from matplotlib.pyplot import imread
from cv2 import resize

converter = tf.lite.TFLiteConverter.from_saved_model("AWS")
model = converter.convert()

with open('AWS.tflite', 'wb') as f:
	f.write(model)

interpreter = tf.lite.Interpreter(model_path = 'AWS.tflite')

interpreter.allocate_tensors()

img = imread('predb/glass458.jpg')
img = resize(img,(256,256))
img = img.reshape((1,256,256,3))
img = img.astype('float32')

interpreter.set_tensor(0, img)

interpreter.invoke()

output_details = interpreter.get_output_details()
output_data = interpreter.get_tensor(output_details[0]['index'])

print(output_data)


