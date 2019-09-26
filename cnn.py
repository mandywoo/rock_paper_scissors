from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras import backend as K

class RPS_CNN:
    @staticmethod
    def build_cnn(width, height, depth, classes):
        model = Sequential()
        input_shape = (height, width, depth)

        channel_dimension = -1

        if K.input_data_format() == 'channels_first':
            input_shape = (depth, height, width)
            channel_dimension = 1

        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pooling_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(classes, activation='softmax'))

        return model