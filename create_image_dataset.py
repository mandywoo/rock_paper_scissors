# use opencv to capture images 
# organize captured images into file paths
# image_dataset -> hand_gesture_1 -> img
#                                 -> img
#               -> hand_gesture_2 -> img


import cv2
from imutils.video import VideoStream
import imutils
import argparse
import time
import os
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='name of image you\'re recording')
ap.add_argument('-n', '--number-of-images', required=True, help='number of images you want saved in the dataset')
ap.add_argument('-p', '--dataset-path', required=True, help='out path for dataset folder to reside in')
args = vars(ap.parse_args())

# make folder for image dataset
dataset_path = args['dataset_path'] + os.sep + args['image']
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# start videostream
vs = VideoStream(src=0).start()
time.sleep(1.0)

# keep track of frame number for naming purposes
frame_count = 0


fgbg = cv2.createBackgroundSubtractorMOG2()



while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=100)

    frame = fgbg.apply(frame, learningRate=0.005)

    cv2.imshow('Frame', frame)

    # save image in correct dataset path
    frame_count += 1
    cv2.imwrite(dataset_path + os.sep + 'img_' + str(frame_count) + '.png', frame)

    # exit conditions
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or frame_count == int(args['number_of_images']):
        break


cv2.destroyAllWindows()
vs.stop()