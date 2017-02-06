from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import cv2

import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.layers import Activation, Merge,Convolution2D, ELU, Flatten, Dropout, Dense, Lambda, MaxPooling2D, LSTM,Reshape
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.wrappers import TimeDistributed
import cv2

def crop_Image(image):
    height, width = image.shape[:2]
    # print (height,width)
    #Remove a region from top and bottom of the image to remove sky and hood
    upper_limit = int((6.0/7.0)*height)
    lower_limit = int((2.0 / 7.0) * height)
    # upper_limit = int((7.0/8.0)*height)
    # lower_limit = int((2.0 / 8.0) * height)
    return image[lower_limit:upper_limit,:]

def preProcess(image):
    image = crop_Image(image)
    image= cv2.resize(image, (200, 66))
    return image



if __name__ == "__main__":
    reader = pd.read_csv('driving_log.csv')

    X_center = reader["center"]
    X_left = reader["left"].map(str.strip)
    X_right = reader["right"].map(str.strip)

    Y_center = reader["steering"]
    Y_left = reader["steering"] + 0.25
    Y_right = reader["steering"] - 0.25

    X_train = X_left
    Y_train = Y_left


    timesteps = 10
    volumesPerBatch = 8
    transformed_image_array = np.zeros((volumesPerBatch, timesteps, 3, 66, 200), dtype="uint8")
    # print ("Entered Here... Final shape of image captured")
    # print (image_array.shape)

    for j in range(volumesPerBatch):
        for k in range(timesteps):
            imPath = X_train[k+j*timesteps]
            image = cv2.imread(imPath)
            image = preProcess(image)
            image_interChangedDimensions = np.transpose(image, (2, 0, 1))
            transformed_image_array[j, k, :, :, :] = image_interChangedDimensions


    print ("transformed_image_array ",transformed_image_array.shape)

    with open("model.json", 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = "model.h5"
    model.load_weights(weights_file)

    steering_angle = model.predict(transformed_image_array,batch_size=volumesPerBatch)
    print ("steering_angle_shape",steering_angle.shape)

