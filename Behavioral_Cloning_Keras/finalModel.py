
# coding: utf-8

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, Lambda, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
import cv2
# print (X_left)
# print (Y_right)
# plt.hist(Y_center)
# plt.title("Steering angles Histogram")
# plt.xlabel("Steering angle")
# plt.ylabel("Frequency")
# plt.show()
# # X_left,Y_left = reader[:,1],reader[:,6]
# X_right,Y_right = reader[:,2],reader[:,6]





def load_images_from_folder(Path):
    # output = np.array([img_to_array(load_img(im)) for im in Path])
    output = []
    output_flipped=[]
    i = 1
    for im in Path:
        print ("loading image : ",i, "  ",im)
        i = i+1
        image=cv2.imread(im)
        print(image.dtype)
        # image = cv2.resize(image, (64, 64))
        # b,g,r = cv2.split(image)           # get b, g, r
        # image = cv2.merge([r,g,b])
        # image = cv2.resize(image, (200, 66))
        # image_flipped = cv2.flip(image,1)
        image_flipped = crop_Image(image)
        image_flipped = cv2.resize(image_flipped, (200, 66))
        print (image_flipped.shape)
        cv2.imshow('image_orig',image)
        cv2.imshow('image_flipped',image_flipped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        image = np.expand_dims(image, axis=0)
        output.append(image)
        output_flipped.append(image_flipped)

    output = np.array(output)
    output_flipped = np.array(output_flipped)
    # output = output.astype('float32')
    # output_flipped = output_flipped.astype('float32')
    print (output.shape)
    print (output_flipped.shape)
    return output,output_flipped

#     print(output.shape)
    

    


def return_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(66, 200, 3)))

    # layer 1 
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same", init = 'he_normal'))
    model.add(ELU())

    # layer 2 
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid",init = 'he_normal'))
    model.add(ELU())
    # model.add(Dropout(.4))
    # model.add(MaxPooling2D((2, 2), border_mode='valid'))

    # layer 3 
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid",init = 'he_normal'))
    model.add(ELU())

    # layer 4
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid",init = 'he_normal'))
    model.add(ELU())
    # model.add(Dropout(.4))

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

    # Finally a single output, since this is a regression problem
    model.add(Dense(1,init = 'he_normal'))

    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss="mse")

    return model



def crop_Image(image):
    height, width = image.shape[:2]
    # print (height,width)
    #Remove a region from top and bottom of the image to remove sky and hood
    upper_limit = int((6.0/7.0)*height)
    lower_limit = int((2.0 / 7.0) * height)
    # upper_limit = int((7.0/8.0)*height)
    # lower_limit = int((2.0 / 8.0) * height)
    return image[lower_limit:upper_limit,:]


def show_image(image):
    cv2.imshow('image_original', image)
    cv2.imshow('image_changed', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def change_brightness(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    brightness = .25+np.random.uniform()
    brightness = round(brightness,3)
    image[:,:,2] = image[:,:,2]*brightness
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

#Input image is a numpy array of an image 
def preProcess(image,steeringAngle,flipImage = False):
    image = crop_Image(image)
    image= cv2.resize(image, (200, 66))
    changeBrightness = np.random.choice([True,False])
    if changeBrightness:
        image = change_brightness(image)
    if flipImage:
        image = cv2.flip(image,1)
        steeringAngle = -1.0 * steeringAngle

    return image,steeringAngle



def batchImageGenerator(X_train,Y_train, batchSize = 64):

    while True:
        X_trainBatchImages = []
        Y_trainBatchImages = []

        for i in range(batchSize):
            index = np.random.randint(0,len(Y_train)-1)
            imPath = X_train[index]
            image=cv2.imread(imPath)
            steeringAngleInitial = Y_train[index]
            if "left" in imPath or 'right' in imPath:
                # print (imPath)
                flipImageChoice = np.random.choice([True, False])
                image, steeringAngle = preProcess(image, steeringAngleInitial,flipImage=flipImageChoice)
            else:
                image,steeringAngle = preProcess(image,steeringAngleInitial)
            X_trainBatchImages.append(image)
            Y_trainBatchImages.append(steeringAngle)

        X_trainBatchImages=np.array(X_trainBatchImages)
        Y_trainBatchImages=np.array(Y_trainBatchImages)


        yield (X_trainBatchImages,Y_trainBatchImages)





if __name__ == "__main__":

    reader=pd.read_csv('driving_log.csv')

    X_center = reader["center"]
    # Strip extra white space
    X_left = reader["left"].map(str.strip)
    X_right = reader["right"].map(str.strip)


    Y_center = reader["steering"]
    Y_left = reader["steering"]+0.25
    Y_right = reader["steering"]-0.25

    X_train = np.hstack((X_center,X_left,X_right))
    Y_train = np.hstack((Y_center,Y_left,Y_right))

    print("X_train ",len(X_train))
    print("Y_train ",len(Y_train))

    # plt.hist(Y_train)
    # plt.title("Steering angles Histogram")
    # plt.xlabel("Steering angle")
    # plt.ylabel("Frequency")
    # plt.show()

    # print (X_center[:2])
    # print (X_left[:2])

    # batch_size = 256
    # nb_epoch = 10
    # print("Loading images ..... ")
    # No_of_images = 6000
    # X_train,X_train_flipped = load_images_from_folder(X_center[:No_of_images])
    # Y_train = Y_center[:No_of_images]
    # Y_train_flipped = -1.0 * Y_train
    #
    # print(Y_train)
    # print (" Break")
    # print(Y_train_flipped)
    #
    # X_validation,X_validation_flipped = load_images_from_folder(X_left[:No_of_images])
    # Y_validation = Y_left[:No_of_images]
    # Y_validation_flipped = -1.0 * Y_validation

    model = return_model()


    # model.fit(X_train, Y_train,batch_size=batch_size,nb_epoch=nb_epoch,validation_data=(X_validation, Y_validation),shuffle=True)
    model.fit_generator(batchImageGenerator(X_train,Y_train,batchSize=256),samples_per_epoch = 256*150, nb_epoch=3)
    model.save_weights('model.h5')
    with open('model.json', 'w') as outfile:
        outfile.write(model.to_json())