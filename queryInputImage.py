# calculate a face embedding for each face in the dataset using facenet
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
import numpy
from keras.models import load_model
import pandas as pd
from extractFacesFeatures import get_embedding
from loaddataset import load_faces
import os
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def compareFeaturesByEuclideanDistance(queryEmbedding):
  df = pd.read_csv('database/csvfiles/file.csv', header=None)
  first_column = df.columns[0]
  df = df.drop([first_column], axis=1)
  print(df)
# load the facenet model
model = load_model('facenet_keras.h5')
print('Loaded Model')
# convert each face in the train set to an embedding
queryPath = "database/testImages/"
face_pixels = load_faces(queryPath)
embedding = get_embedding(model, face_pixels[0])
# newTrainX.append(embedding)
# newTrainX = asarray(newTrainX)
compareFeaturesByEuclideanDistance(embedding)
queryImageEmbedding = asarray(embedding)

print(queryImageEmbedding.shape)
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