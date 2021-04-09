from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from detectface import extract_face, detectFace, load_faces

# load images and extract faces for all images in a directory
def load_faces(directory):
	faces = list()
	# enumerate files
	for filename in listdir(directory):
		# path
		path = directory + filename
		# get face
		face = extract_face(path)
		# store
		faces.append(face)
	return faces
 
# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		# y.extend(labels)
    # loại bỏ return asarray(y)
	return asarray(X)
 
def saveFaceDataset(path, file):
  # load train dataset
  # trainX, trainy = load_dataset('archive/train/')
  trainX = load_dataset(path)
  print(trainX)
  print(trainX.shape)
path = 'database/images/'
file = 'harrypotter.npz'
saveFaceDataset(path, file)
  # load test dataset
  # testX, testy = load_dataset('archive/val/')
  # save arrays to one file in compressed format
  #       ===============================
  # savez_compressed('harrypotter.npz', trainX)
  #savez_compressed(file, trainX)