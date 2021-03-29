import os
import logging
import tensorflow as tf
from numpy import load
from detectface import extract_face, detectFace, load_faces
tf.get_logger().setLevel(logging.ERROR)
# example of loading the keras facenet model
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# load the model
model = load_model('facenet_keras.h5', compile=False)
# summarize input and output shape
print(model.inputs)
print(model.outputs)
path = 'database/images/'
#face = extract_face(path)
#print(i, face.shape)
# face = load_faces(path)
# print(face)
#data = load(path)
#print(data)
detectFace(path)
print('nam')