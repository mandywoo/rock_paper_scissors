# so figures can be saved in background
import matplotlib
matplotlib.use('Agg')

from keras.preprocessing.image import ImageDataGenerator   # for data augmentation
from keras.optimizers import Adam   # optimizer used to train network
from keras.preprocessing.image import img_to_array
# allows us to input set of class labels, transform labels into one-hot encoded vectors, 
# then allow us to take an integer class label prediction from Keras CNN and transform it back into a human-readable label
from sklearn.preprocessing import LabelBinarizer    
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from cnn import RPS_CNN
from batch_size import FindBatchSize

EPOCHS = 25    # num of epochs to train for (how many times the network sees each training example and learns patterns from it)
INIT_LR = 1e-3  # initial learning rate (1e-3 is default for Adam optimizer)
BS = FindBatchSize(RPS_CNN)      # batch size (we will pass batches of images into the network for training)
IMAGE_DIMS = (100, 100, 3)    # image dimensions (96x96 pixels, 3 channels)