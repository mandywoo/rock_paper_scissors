# start video stream
# ask play game?
# thumbs up = yes
# start countdown 3-2-1
# rock - paper - scissor
# computer is random for now -> later implement self learning
# display winner
# restart
# add hand motion to stop game

# dataset collect: rock, paper, scissor, thumbs up, x
# look into imagedatagenerator


from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import random
from imutils import paths
from imutils.video import VideoStream
import time

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', help='path to saved model')
ap.add_argument('-l', '--labels', help='path to saved binarized labels')
args = vars(ap.parse_args())

# load network and labels
print('[INFO] loading network...')
model = load_model(args['model'])
lb = pickle.loads(open(args['labels'], 'rb').read())

print('[INFO] starting videostream...')
vs = VideoStream(src=0).start()
time.sleep(1.0)

fgbg = cv2.createBackgroundSubtractorMOG2()

moves = ['rock', 'paper', 'scissor']
game_rule = {'rock': 'scissor', 'paper': 'rock', 'scissor': 'paper'}

while True:
    frame = vs.read()

    frame = imutils.resize(frame, width=600)

    # add timer and condition to start game and display countdown


    aug_frame = fgbg.apply(frame, learningRate=0.005)

    # classifying input image
    probabilities = model.predict(aug_frame)[0]
    index = np.argmax(proba)       # argmax: Returns the indices of the maximum values along an axis.
    label = lb.classes_[index]

    # build and draw label on image
    label = '{}: {:.2f}%'.format(label, probabilities[index])
    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # random robot move -----> to be changed
    robot_move = random.choice(moves)

    # compare robot move to human move
    win = None
    if game_rule[robot_move] == label:
        win = 'robot'
    elif robot_move == label:
        win = 'tie'
    else:
        win = 'human'

    win_text = 'Robot: {} vs. Human: {} {}'.format(robot_move, label, win)
    cv2.putText(frame, win_text, (100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    # show frame
    print('[INFO] {}: {:.2f}%'.format(label, probabilities[index]))
    cv2.imshow('Game', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cv2.destroyAllWindows()
vs.stop()