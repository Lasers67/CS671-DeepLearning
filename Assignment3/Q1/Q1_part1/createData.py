import numpy as np
import os
from PIL import Image
import cv2
from random import shuffle
def getDataset(dataPath,split=0.75,save=False):
	train=[]
	test=[]
	classFolders=["Knuckle","Palm","Vein"]
	# classFolders=["Knuckle"]
	maxShape=[0,0]
	for i in range(len(classFolders)):
		fileRead=0
		oneClass=[]
		path=dataPath+classFolders[i]+'/'
		content=[[x for x in line.split(',')] for line in open(path+'groundtruth.txt')]
		for line in content:
			if(fileRead==200):
				break
			arr = np.asarray(Image.open(path+line[0]))
			if(arr.shape[0]>maxShape[0]):
				maxShape[0] = arr.shape[0]
			if(arr.shape[1]>maxShape[1]):
				maxShape[1] = arr.shape[1]
			oneClass.append([arr,[int(line[1]),int(line[2]),int(line[3]),int(line[4])],i])
			fileRead+=1
		shuffle(oneClass)
		train.append(oneClass[:int(len(oneClass)*split)])
		test.append(oneClass[int(len(oneClass)*split):])
	train = np.asarray(train)
	test = np.asarray(test)
	train = train.reshape(-1,train.shape[-1])
	test = test.reshape(-1,test.shape[-1])
	shuffle(train)
	shuffle(test)
	trainData = []
	testData = []
	train_bbox = []
	test_bbox = []
	train_class = []
	test_class = []
	for x in train:
		trainData.append(x[0])
		train_bbox.append(x[1])
		oneHot=[0,0,0]
		oneHot[x[2]]=1
		train_class.append(oneHot)
	for x in test:
		testData.append(x[0])
		test_bbox.append(x[1])
		oneHot=[0,0,0]
		oneHot[x[2]]=1
		test_class.append(oneHot)
	maxShapeArr = np.zeros(maxShape)
	for i in range(len(testData)):
		# print(test[i][0][1][2])
		tempArr = maxShapeArr

		tempArr[:testData[i].shape[0],:testData[i].shape[1]]=testData[i]
		testData[i] = tempArr
		# print(test[i][0][1][2])
	for i in range(len(trainData)):
		# print(train[i][0][1][2])
		tempArr = maxShapeArr
		tempArr[:trainData[i].shape[0],:trainData[i].shape[1]]=trainData[i]
		trainData[i] = tempArr
		# print(test[i][0][1][2])
	trainData = np.asarray(trainData)
	testData = np.asarray(testData)
	trainData = trainData.reshape((trainData.shape[0],trainData.shape[1],trainData.shape[2],1))
	testData = testData.reshape((testData.shape[0],testData.shape[1],testData.shape[2],1))
	train_bbox = np.asarray(train_bbox)
	test_bbox = np.asarray(test_bbox)
	train_class = np.asarray(train_class)
	test_class = np.asarray(test_class)
	# if(save):
	# 	np.save('train.py',train)
	# 	np.save('test.py',test)
	return trainData,testData,train_bbox,test_bbox,train_class,test_class