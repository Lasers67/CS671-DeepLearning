import numpy as np 
import os
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.models import model_from_json
from PIL import Image
json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)

model.load_weights('model/weights.hdf5')


print ("Model Loaded")
testFolder = "Test/"
saveFolder = "Output/"

for data_file in sorted(os.listdir(testFolder)):
  image = Image.open(testFolder + data_file)
  image = image.resize((400, 288),  Image.ANTIALIAS) 
  y_pred = model.predict(np.array([np.array(image)/255.]))
  y_pred = np.reshape(y_pred[0], (288,400))
  y_pred[y_pred<=0.5] = 0
  y_pred[y_pred>0.5] = 1

  # print (y_pred[0].shape)
  image = Image.fromarray(y_pred*255).convert('L')
  image = image.resize((400, 300),  Image.ANTIALIAS) 
  # image.show()
  image.save(saveFolder + data_file + '.png')
