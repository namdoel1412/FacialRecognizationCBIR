# calculate a face embedding for each face in the dataset using facenet
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
import numpy
from keras.models import load_model
import pandas as pd

# lấy face embedding cho mỗi ảnh mặt người
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]
 
# load the face dataset
def saveFacialFeaturesImages(npzFile, dataFilePath):
  data = load(npzFile)
  # gán trainx với array faceembbeding
  trainX = data['arr_0']
  # print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
  print('Loaded: ', trainX.shape)
  # load the facenet model
  model = load_model('facenet_keras.h5')
  print('Loaded Model')
  # convert each face in the train set to an embedding
  newTrainX = list()
  for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
  newTrainX = asarray(newTrainX)
  print(newTrainX.shape)
  print(newTrainX)
  # numpy.savetxt("database/foo.csv", newTrainX, delimiter=",")
  # ---------------------------
  # pd.DataFrame(newTrainX).to_csv("database/csvfiles/file.csv")
  pd.DataFrame(newTrainX).to_csv(dataFilePath)
  # ---------------------------
  # convert each face in the test set to an embedding
  # newTestX = list()
  # for face_pixels in testX:
  # 	embedding = get_embedding(model, face_pixels)
  # 	newTestX.append(embedding)
  # newTestX = asarray(newTestX)
  # print(newTestX.shape)