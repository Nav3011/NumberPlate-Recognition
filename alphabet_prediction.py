from PIL import Image
import numpy as np
from keras.models import load_model
from keras.models import model_from_json
# model = load_model('model.h5')


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print("Loaded model from disk")

img = Image.open("sample.png").convert("L")
im2arr = np.array(img.resize((28,28)))
input_img = im2arr.reshape(1,28,28,1)
y_pred = loaded_model.predict(input_img)
print(y_pred)
