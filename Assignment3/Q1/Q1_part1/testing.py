import os
import numpy as np
import cv2
from PIL import Image
from keras.models import Model
from keras.models import model_from_json
from IntersectionOverUnion import bb_intersection_over_union
import sys

arguments = sys.argv[1:]
num_args = len(arguments)
if num_args < 3:
	print("Usage: python testing.py <ImageFolderLocation> <groundTruthFileLocation> <OutputFileName>")
	exit()

imagesLocation = arguments[0]
groundTruthFileLocation = arguments[1]
OutputFileName = arguments[2]
shape = (1,480,640,1)
file = open(OutputFileName,"w")
# print(imagesLocation,groundTruthFileLocation,OutputFileName)
json_file = open("model.json","r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("part1_joint_model_test.h5")
content=[[x for x in line.split(',')] for line in open(groundTruthFileLocation)]
for line in content:
	arr = np.asarray(Image.open(imagesLocation+line[0]))
	# arr = cv2.imread(imagesLocation+line[0])
	# cv2.imshow('try',arr)
	# cv2.waitKey(0)
	arr = cv2.resize(arr,(shape[1],shape[2]))
	arr = arr.reshape(shape)
	gt_bbox = [int(line[1]),int(line[2]),int(line[3]),int(line[4])]
	predicted_box = loaded_model.predict(arr)[0][0]
	iou = bb_intersection_over_union(predicted_box,gt_bbox)
	file.write(line[0]+' '+str(iou)+'\n')