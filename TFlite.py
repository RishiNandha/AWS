import tensorflow as tf

con = tf.lite.TFLiteConverter.from_saved_model("AWS")
model = converter.convert()

with open('AWS.tflite', 'wb') as f:
	f.write(model)

i = tf.lite.Interpreter(model_path = 'AWS.tflite')
i.allocate_tensors()

