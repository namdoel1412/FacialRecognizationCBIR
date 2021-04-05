from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from loaddataset import saveFaceDataset
from extractFacesFeatures import saveFacialFeaturesImages

path = 'database/images/'
file = 'harrypotter.npz'
featureFile = 'database/csvfiles/harry_.csv'
saveFaceDataset(path, file)
saveFacialFeaturesImages(file, featureFile)