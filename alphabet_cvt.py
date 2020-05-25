# Improting Image class from PIL module 
from PIL import Image 
import PIL.ImageOps
import numpy as np
from keras.models import load_model
from keras.models import model_from_json
# Opens a image in RGB mode 
im = Image.open(r"thresh_plate.jpg") 

json_file_alphabet = open('alphabet_model.json', 'r')
loaded_model_json_alphabet = json_file_alphabet.read()
json_file_alphabet.close()
loaded_model_alphabet = model_from_json(loaded_model_json_alphabet)
loaded_model_alphabet.load_weights("alphabet_model.h5")
loaded_model_alphabet.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print("Loaded alphabet model from disk")

# Size of the image in pixels (size of orginal image) 
# (This is not mandatory) 
width, height = im.size 
  
# Setting the points for cropped image 
# left = 5
# top = height / 4
# right = 164
# bottom = 3 * height / 4
  
# Cropped image of above dimension 
# (It will not change orginal image) 
# im1 = im.crop((left, top, right, bottom)) 
verticals = [0, 40, 82, 123, 164, 206, 248, 290, 332, 373, 415]
for i in range(len(verticals)-1):
	left = verticals[i]
	top = 0
	right = verticals[i+1]
	bottom = height
	im1 = im.crop((left, top, right, bottom))
	inverted_image = PIL.ImageOps.invert(im1)

	# img1 = Image.open("alphabet.png").convert("L")
	im2arr1 = np.array(inverted_image.resize((28,28)))
	input_img1 = im2arr1.reshape(1,28,28,1)
	y_pred1 = loaded_model_alphabet.predict(input_img1)
	# print(np.argmax(y_pred1))
	print(chr(97+np.argmax(y_pred1)).upper(),end=" ")
	# print(type(im1))
# Shows the image in image viewer 
	# inverted_image.show()