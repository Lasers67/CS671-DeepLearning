import os
import numpy as np
from random import shuffle
from PIL import Image
import createData
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Activation, BatchNormalization,Dropout
from keras.models import Model
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam
# import IntersectionOverUnion

trainData,testData,train_bbox,test_bbox,train_classification,test_classification = createData.getDataset("../Q1/",split=0.95)
shape=(trainData[0].shape[0],trainData[0].shape[1],1)
input_layer=Input(shape)
conv1 = Conv2D(64, kernel_size=(10,10),strides=5, activation="relu",input_shape=shape)(input_layer)
conv1=BatchNormalization()(conv1)
pool1 = MaxPooling2D(pool_size=(3,3), strides=2)(conv1)
pool1=Dropout(0.4)(pool1)
flat = Flatten()(pool1)
dense1 = Dense(64)(flat)
dense2 = Dense(64,activation='relu')(flat)
dense2=Dropout(0.4)(dense2)
dense_bbox_layer = Dense(4,name="final_bbox_layer")(dense1)
dense_class_layer = Dense(3,activation="softmax",name="final_class_layer")(dense2)
model = Model(inputs=input_layer,outputs=[dense_bbox_layer,dense_class_layer])
model.summary()
losses = {
	"final_bbox_layer":"mse",
  "final_class_layer": "categorical_crossentropy"
}
model.compile(loss=losses,optimizer=Adam(lr=1e-5),metrics=["accuracy"])
model.fit(trainData,{
	"final_bbox_layer":train_bbox,
  "final_class_layer":train_class
},epochs=15, shuffle=True, batch_size=128)
model.save('part1_joint_model_test.h5')
# model.evaluate(testData,{
# 	"final_layer":test_bbox,
#   "final_"
# })