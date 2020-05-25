from PIL import Image
import numpy as np
from keras.models import load_model
from keras.models import model_from_json
# model = load_model('model.h5')


json_file_alphabet = open('alphabet_model.json', 'r')
loaded_model_json_alphabet = json_file_alphabet.read()
json_file_alphabet.close()
loaded_model_alphabet = model_from_json(loaded_model_json_alphabet)
loaded_model_alphabet.load_weights("alphabet_model.h5")
loaded_model_alphabet.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print("Loaded alphabet model from disk")

img1 = Image.open("alphabet.png").convert("L")
im2arr1 = np.array(img1.resize((28,28)))
input_img1 = im2arr1.reshape(1,28,28,1)
y_pred1 = loaded_model_alphabet.predict(input_img1)
print(y_pred1)

json_file_digit = open('digit_model.json', 'r')
loaded_model_json_digit = json_file_digit.read()
json_file_digit.close()
loaded_model_digit = model_from_json(loaded_model_json_digit)
loaded_model_digit.load_weights("digit_model.h5")
loaded_model_digit.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print("Loaded digit model from disk")


# model = load_model('model_digit.h5')
# img = Image.open("number.png").convert("L")
# im2arr = np.array(img.resize((28,28)))
# input_img = im2arr.reshape(1,28,28,1)
# y_pred = model.predict(input_img)
# print(y_pred)
img2 = Image.open("number.png").convert("L")
im2arr2 = np.array(img2.resize((28,28)))
input_img2 = im2arr2.reshape(1,28,28,1)
y_pred2 = loaded_model_digit.predict(input_img1)
print(y_pred2)





