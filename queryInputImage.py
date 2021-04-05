# calculate a face embedding for each face in the dataset using facenet
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
import numpy
from keras.models import load_model
import pandas as pd
from extractFacesFeatures import get_embedding, saveFacialFeaturesImages
from loaddataset import load_faces
import math
import os
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def compareFeaturesByEuclideanDistance(queryEmbedding):
  df = pd.read_csv('database/csvfiles/harry_.csv', header=None)
# lấy column đầu tiên
  first_column = df.columns[0]
# xóa column đầu tiên
  df = df.drop([first_column], axis=1)
  df = df.iloc[1:]
  # print(df)
  arr = df.to_numpy()
  # print(arr)
# xóa row đầu tiên
  #df = df.drop(index=df.index[0], axis=0, inplace=True)
  minn = math.inf
  res = list()
  print(arr)
  for x in arr:
    # print(x)
    # print(asarray(queryEmbedding))
    dist = numpy.linalg.norm(x - asarray(queryEmbedding))
    res.append(abs(dist))
    if minn > abs(dist):
      minn = abs(dist)
  #print(df)
  print(res)
  print(minn)
# load the facenet model
model = load_model('facenet_keras.h5')
print('Loaded Model')
# convert each face in the train set to an embedding
queryPath = "database/testImages/"
face_pixels = load_faces(queryPath)
embedding = get_embedding(model, face_pixels[0])
# newTrainX.append(embedding)
# newTrainX = asarray(newTrainX)
queryImageEmbedding = asarray(embedding)
compareFeaturesByEuclideanDistance(queryImageEmbedding)
print(queryImageEmbedding)
# print(queryImageEmbedding)
# numpy.savetxt("database/foo.csv", newTrainX, delimiter=",")
# pd.DataFrame(newTrainX).to_csv("database/csvfiles/file.csv")
# convert each face in the test set to an embedding
# newTestX = list()
# for face_pixels in testX:
# 	embedding = get_embedding(model, face_pixels)
# 	newTestX.append(embedding)
# newTestX = asarray(newTestX)
# print(newTestX.shape)