import numpy as np
# from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Activation, BatchNormalization
# from keras.models import Model
# from tensorflow.examples.tutorials.mnist import input_data
# from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg 
from PIL import Image

trainPath = "Data/Data/"
truthPath = "Data/Mask/"
c = 1
img = []
for data_file in sorted(os.listdir(trainPath)):
	image = Image.open(trainPath + data_file)
	image = image.resize((400, 288),  Image.ANTIALIAS) 
	img.append(np.array(image))
	print(c)
	c+=1
img = np.asarray(img)
print img.shape
np.save('train.npy', img)

img = []
for data_file in sorted(os.listdir(truthPath)):
	image = Image.open(truthPath + data_file)
	image = image.resize((400, 288),  Image.ANTIALIAS) 
	img.append(np.array(image))
	print(c)
	c+=1
img = np.asarray(img)
print img.shape
np.save('gt.npy', img)

