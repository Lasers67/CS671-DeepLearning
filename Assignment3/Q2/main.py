import numpy as np

##Load Data

train = np.load('gdrive1/My Drive/Data/train_scale.npy')
gt = np.load('gdrive1/My Drive/Data/gt_scale.npy')


import numpy as np 
import os
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


input_size = (288,400,3)
inputs = Input(input_size)
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)

up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
merge6 = concatenate([drop4,up6], axis = 3)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
merge7 = concatenate([conv3,up7], axis = 3)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
merge8 = concatenate([conv2,up8], axis = 3)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
merge9 = concatenate([conv1,up9], axis = 3)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

model = Model(input = inputs, output = conv10)

model.compile(optimizer = Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
from keras.utils import plot_model
plot_model(model, to_file='model.png')

def trainGenerator(batch_size=6):
  global train, gt
  randomize = np.arange(8000)
  np.random.shuffle(randomize)
  train1 = train[randomize]
  gt1 = gt[randomize]
  k = 0
  while 1:
    X,y = train1[k:k+batch_size],gt1[k:k+batch_size]
    k+=batch_size
    if(k>8000): k =0
    X = X/255.
    y = y/255.
    y[y>0.5] = 1
    y[y<=0.5] = 0
    y = np.reshape(y, y.shape + (1,))
    yield (X,y)

myGene = trainGenerator()
model_checkpoint = ModelCheckpoint('weights.hdf5',verbose=1)
# model.load_weights('weights.hdf5')
model.fit_generator(myGene,steps_per_epoch=500,epochs=3, callbacks=[model_checkpoint])

# 500/500 [==============================] - 451s 902ms/step - loss: 0.1026 - acc: 0.9564


# y_pred = model.predict(np.array([train[9500]/255.]))
# # from PIL import Image
# # import matplotlib.pyplot as plt
# # k = np.reshape(y_pred[0], (288,400))
# # l = np.reshape(k,(288*400))
# # print (set(l), len(set(l)))
# # k[k<=0.5] = 0
# # k[k>0.5] = 1
# # l = np.reshape(k,(288*400))
# # print (set(l), len(set(l)))
# # im = Image.fromarray(k)
# # # im.show()
# # plt.imshow(im)
# # plt.show()
# # im = Image.fromarray(gt[9500])
# # plt.imshow(im)
# # plt.show()
# # im = Image.fromarray(train[9500])
# # plt.imshow(im)
# # plt.show()

