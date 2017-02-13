
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, Lambda
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import cv2

# Always generate fixed random numbers.
initial_seed = 2016
np.random.seed(initial_seed)

#  The model that is used to train the CNN.
# Implemented the Nvidia architecture
def return_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(66, 200, 3)))

    # layer 1 
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same", init = 'he_normal'))
    model.add(ELU())

    # layer 2 
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid",init = 'he_normal'))
    model.add(ELU())

    # layer 3 
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid",init = 'he_normal'))
    model.add(ELU())

    # layer 4
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid",init = 'he_normal'))
    model.add(ELU())

    # layer 5
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid",init = 'he_normal'))
    model.add(ELU())

    # Flatten the output
    model.add(Flatten())

    # layer 4
    model.add(Dense(1164,init = 'he_normal'))
    model.add(Dropout(.4))
    model.add(ELU())

    # layer 5
    model.add(Dense(100,init = 'he_normal'))
    model.add(Dropout(.2))
    model.add(ELU())

    # layer 6
    model.add(Dense(50,init = 'he_normal'))
    model.add(ELU())

    # layer 7
    model.add(Dense(10,init = 'he_normal'))
    model.add(ELU())

    # A single output value, as we need to predict steering angle.
    model.add(Dense(1,init = 'he_normal'))

    # Use adam optimizer with learning rate of 0.0001
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss="mse")

    return model

def create_shadows(image):
    nrows,ncols = image.shape[:2]
    channel_count = image.shape[2]
    # print(nrows,ncols,image.dtype)
    mask = np.zeros(image.shape,dtype = np.uint8)
    startShadowLeft = np.random.choice([True,False])
    if startShadowLeft:
        col = 0
    else:
        col = ncols
    shadowColStarts = np.random.random_integers(5,ncols-5,2)
    roi_corners = np.array([[(col, 0), (col,ncols), ( shadowColStarts[1],ncols),(shadowColStarts[0],0)]], dtype=np.int32)
    OldRange = (1.0 - 0.0)
    NewRange = (0.9 - 0.3)
    OldValue = np.random.sample()
    random_brightness_to_multiply = (((OldValue - 0) * NewRange) / OldRange) + 0.3
    random_brightness_to_multiply = round(random_brightness_to_multiply,3)
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    coordinatesROI = mask==255
    image[coordinatesROI] = image[coordinatesROI]*random_brightness_to_multiply
    return image

# Cropping function used to crop the hood of the car and part of the sky.
def crop_Image(image):
    height, width = image.shape[:2]
    #Remove a region from top and bottom of the image to remove sky and hood
    lower_limit = int((3.0/10.0) * height)
    upper_limit = int((8.5/10.0)*height)
    return image[lower_limit:upper_limit,:]

# Change brightness of an image multiplying with a random value in HSV space.
def change_brightness(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    brightness = .25+np.random.uniform()
    brightness = round(brightness,3)
    image[:,:,2] = image[:,:,2]*brightness
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

#Apply image preprocessing on an image.
#Crop the image, apply brightness and flipping randomly.
def preProcess(image,steeringAngle,flipImage = False):
    image = crop_Image(image)
    image= cv2.resize(image, (200, 66))
    changeBrightness = np.random.choice([True,False])
    # createShadow = np.random.choice([True,False])
    if changeBrightness:
        image = change_brightness(image)
    # elif createShadow:
    #     image = create_shadows(image)
    if flipImage:
        image = cv2.flip(image,1)
        steeringAngle = -1.0 * steeringAngle

    return image,steeringAngle

# Generator that generates a random batch of images by applying random preprocessing to images.
def batchImageGenerator(X_train,Y_train, batchSize = 64):
    while True:
        X_trainBatchImages = []
        Y_trainBatchImages = []

        for i in range(batchSize):
            retry = True
            while retry:
                index = np.random.randint(0,len(Y_train)-1)
                imPath = X_train[index]
                image=cv2.imread(imPath)
                steeringAngleInitial = Y_train[index]
                if abs(steeringAngleInitial) < 0.2:
                    retryProbability = np.random.sample()
                    if retryProbability > 0.6:
                        retry = False
                    else:
                        retry = True
                else:
                    retry = False

            if "left" in imPath or 'right' in imPath:
                flipImageChoice = np.random.choice([True, False])
                image, steeringAngle = preProcess(image, steeringAngleInitial,flipImage=flipImageChoice)
            else:
                image,steeringAngle = preProcess(image,steeringAngleInitial)
            X_trainBatchImages.append(image)
            Y_trainBatchImages.append(steeringAngle)

        X_trainBatchImages=np.array(X_trainBatchImages)
        Y_trainBatchImages=np.array(Y_trainBatchImages)

        # print(Y_trainBatchImages)
        yield (X_trainBatchImages,Y_trainBatchImages)


def batchImageGenerator_ValidationData_sequential(X_train,Y_train, batchSize = 64):
    index = 0
    while True:
        if batchSize > len(Y_train):
            raise ValueError("Batch size should be less than the size of input data.")
        if ((index + batchSize) >= len(Y_train)):
            index = 0
        X_trainBatchImages = []
        Y_trainBatchImages = []

        for i in range(batchSize):
            imPath = X_train[index]
            image=cv2.imread(imPath)
            steeringAngle = Y_train[index]
            image = crop_Image(image)
            image = cv2.resize(image, (200, 66))
            index = index + 1
            X_trainBatchImages.append(image)
            Y_trainBatchImages.append(steeringAngle)

        X_trainBatchImages=np.array(X_trainBatchImages)
        Y_trainBatchImages=np.array(Y_trainBatchImages)

        yield (X_trainBatchImages,Y_trainBatchImages)

def batch_validation_fixed(X_train,Y_train,batchSize):
    batchSize = len(Y_train)
    X_trainBatchImages = []
    Y_trainBatchImages = []
    for i in range(batchSize):
        imPath = X_train[i]
        image = cv2.imread(imPath)
        steeringAngle = Y_train[i]
        image = crop_Image(image)
        image = cv2.resize(image, (200, 66))
        X_trainBatchImages.append(image)
        Y_trainBatchImages.append(steeringAngle)

    X_trainBatchImages = np.array(X_trainBatchImages)
    Y_trainBatchImages = np.array(Y_trainBatchImages)

    return (X_trainBatchImages,Y_trainBatchImages)




if __name__ == "__main__":

    reader=pd.read_csv('driving_log.csv')
    # Read the path of the images from the csv file.
    X_center = reader["center"]
    # Strip extra white space
    X_left = reader["left"].map(str.strip)
    X_right = reader["right"].map(str.strip)

    Y_center = reader["steering"]
    # Add an offset for the left and right steering angles
    Y_left = reader["steering"]+0.275
    Y_right = reader["steering"]-0.275

    #  Combine left, center and right image paths
    X_train = np.hstack((X_center,X_left,X_right))
    Y_train = np.hstack((Y_center,Y_left,Y_right))

    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.10,random_state=initial_seed)

    print("Y_train ",len(Y_train))
    print("Y_validation ",len(Y_validation))

    # plt.subplot(2, 1, 1)
    # plt.hist(Y_train)
    # plt.title('Steering angles')
    # plt.ylabel('Y_train')
    #
    # plt.subplot(2, 1, 2)
    # # plt.plot(Y_test, 'r.-')
    # plt.hist(Y_validation)
    # plt.xlabel('Samples')
    # plt.ylabel('Y_validation')
    #
    # plt.show()



    model = return_model()

    #  Use fit generator to generate images in batches.
    validation_loss = 10000.0
    Best_epoch = -1
    Best_validation_loss = 10000.0
    No_of_epochs = 18
    # x_data_validation , y_data_validation = batch_validation_fixed(X_validation,Y_validation,len(Y_validation))

    for i in range(No_of_epochs):
        # np.random.seed(initial_seed+i)
        hist = model.fit_generator(generator = batchImageGenerator(X_train,Y_train,batchSize=256),samples_per_epoch = 128*200, nb_epoch=1 , validation_data=batchImageGenerator_ValidationData_sequential(X_validation,Y_validation,batchSize=2400),nb_val_samples = 2400)
        print ("Validation_Loss: " ,hist.history['val_loss'], "epoch: ",i)
        # hist = model.fit_generator(generator=batchImageGenerator(X_train, Y_train, batchSize=10),samples_per_epoch=10, nb_epoch=1)
        # From the way loss is calculated it is understood that it is non deterministic in the background
        # So we cannot produce reproducible results with tensorflow backend.
        # print("Loss: ", hist.history['loss'], "epoch: ", i)
        # current_epoch_loss = model.evaluate(x_data_validation,y_data_validation,batch_size=len(Y_validation))
        current_epoch_loss = float(hist.history['val_loss'][0])
        if current_epoch_loss < validation_loss:
            print("Saving model with Validation_Loss: ", current_epoch_loss, " at epoch: ", i)
            Best_epoch = i
            Best_validation_loss = current_epoch_loss
            model.save_weights('model.h5')
            with open('model.json', 'w') as outfile:
                outfile.write(model.to_json())
            validation_loss = current_epoch_loss

    print (" Best Epoch : ",Best_epoch, "  Best Validation Loss",Best_validation_loss)