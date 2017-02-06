
# coding: utf-8

# In[1]:
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
    

    
def buildModel(volumesPerBatch, timesteps, cameraFormat=(3, 66, 200)):
  """
  Build and return a CNN + LSTM model; details in the comments.
  The model expects batch_input_shape =
  (volumes per batch, timesteps per volume, (camera format 3-tuple))
  A "volume" is a video frame data struct extended in the time dimension.
  Args:
    volumesPerBatch: (int) batch size / timesteps
    timesteps: (int) Number of timesteps per volume.
    cameraFormat: (3-tuple) Ints to specify the input dimensions (color
        channels, height, width).
    verbosity: (int) Print model config.
  Returns:
    A compiled Keras model.
  """
  print ("Building model...")
  ch, row, col = cameraFormat

  model = Sequential()

  if timesteps == 1:
      raise ValueError("Not supported w/ TimeDistributed layers")

  # Use a lambda layer to normalize the input data
  # It's necessary to specify batch_input_shape in the first layer in order to
  # have stateful recurrent layers later
  model.add(Lambda(
      lambda x: x / 127.5 - 1.,
      batch_input_shape=(volumesPerBatch, timesteps, ch, row, col),
  )
  )

  # For CNN layers, weights are initialized with Gaussian scaled by fan-in and
  # activation is via ReLU units; this is current best practice (He et al., 2014)

  # Several convolutional layers, each followed by ELU activation
  # 8x8 convolution (kernel) with 4x4 stride over 16 output filters
  model.add(TimeDistributed(
      Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", init="he_normal")))
  model.add(Activation("relu"))
  # 5x5 convolution (kernel) with 2x2 stride over 32 output filters
  model.add(TimeDistributed(
      Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", init="he_normal")))
  model.add(Activation("relu"))
  # 5x5 convolution (kernel) with 2x2 stride over 64 output filters
  model.add(TimeDistributed(
      Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same", init="he_normal")))
  # TODO: Add a max pooling layer?

  print("Before Flatten ", model.output_shape)
  # Flatten the input to the next layer; output shape = (None, 76800)
  model.add(TimeDistributed(Flatten()))
  print("After Flatten ", model.output_shape)
  # Apply dropout to reduce overfitting
  # model.add(Dropout(.2))
  # model.add(Activation("relu"))

  # Fully connected layer
  # model.add(TimeDistributed(Dense(512)))
  # # More dropout
  # model.add(Dropout(.2))
  # model.add(Activation("relu"))

  # Add stacked (stateful) LSTM layers
  model.add(LSTM(512,
                 return_sequences=True,
                 batch_input_shape=(volumesPerBatch, timesteps, 320),
                 stateful=True))
  model.add(LSTM(512,
                 return_sequences=True,
                 stateful=True))

  # Fully connected layer with one output dimension (representing the predicted
  # value).
  model.add(TimeDistributed(Dense(1)))

  print("Y output shape ", model.output_shape)

  # Adam optimizer is a standard, efficient SGD optimization method
  # Loss function is mean squared error, standard for regression problems
  model.compile(optimizer="adam", loss="mse")
  return model


def test_model(volumesPerBatch, timesteps, cameraFormat=(3, 66, 200)):
    Izda = Sequential()
    Izda.add(TimeDistributed(Convolution2D(40, 3, 3, border_mode='same'), input_shape=(timesteps, 3, 66, 200)))
    Izda.add(Activation('relu'))
    Izda.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    Izda.add(Dropout(0.2))

    Dcha = Sequential()
    Dcha.add(TimeDistributed(Convolution2D(40, 3, 3, border_mode='same'), input_shape=(timesteps, 3, 66, 200)))
    Dcha.add(Activation('relu'))
    Dcha.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    Dcha.add(Dropout(0.2))

    Frt = Sequential()
    Frt.add(TimeDistributed(Convolution2D(40, 3, 3, border_mode='same'), input_shape=(timesteps, 3, 66, 200)))
    Frt.add(Activation('relu'))
    Frt.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    Frt.add(Dropout(0.2))
    # print (Frt.output_shape)

    merged = Merge([Izda, Dcha, Frt], mode='concat', concat_axis=2)
    # print(merged.output_shape)
    # Output from merge is (batch_size, sequence_length, 120, 4, 5)
    # We want to get this down to (batch_size, sequence_length, 120*4*5)

    model = Sequential()
    model.add(merged)
    print("Before Flatten ", model.output_shape)
    model.add(TimeDistributed(Flatten()))
    print("After Flatten ", model.output_shape)
    model.add(LSTM(240, return_sequences=True))

    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')

    return model


def old_model():
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
    # roi_corners = np.array([[(0, col), (nrows, col), (nrows, shadowColStarts[1]),(0,shadowColStarts[0])]])
    roi_corners = np.array([[(col, 0), (col,ncols), ( shadowColStarts[1],ncols),(shadowColStarts[0],0)]], dtype=np.int32)
    random_brightness_to_multiply = random.uniform(0.25,0.90)
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    coordinatesROI = mask==255
    image[coordinatesROI] = image[coordinatesROI]*random_brightness_to_multiply
    return image
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



#Input image is a numpy array of an image 
def preProcess(image,steeringAngle,flipImage = False):
    image = crop_Image(image)
    image= cv2.resize(image, (200, 66))
    # changeBrightness = np.random.choice([True,False])
    # createShadows = np.random.choice([True,False])
    # if changeBrightness:
    #     image = change_brightness(image)
    # elif createShadows:
    #     image = create_shadows(image)
    # if flipImage:
    #     image = cv2.flip(image,1)
    #     steeringAngle = -1.0 * steeringAngle

    return image,steeringAngle



def batchImageGenerator(X_train,Y_train, batchSize,timesteps):
    index = 0
    while True:
        if batchSize%timesteps:
            raise ValueError("Batch size should be divisible by timesteps so we get an equal number of frames in each portion of the batch.")
        volumesPerBatch = int(batchSize/timesteps)
        # If batchsize makes the current iteration to cross the max length of input array, reset counter.
        if((index+batchSize) >= len(Y_train)):
            index = 0
        X_trainBatchImages =  np.zeros((volumesPerBatch, timesteps, 3, 66, 200), dtype="uint8")
        Y_trainBatchImages = np.zeros((volumesPerBatch, timesteps, 1), dtype="float32")


        for j in range(volumesPerBatch):
            for k in range(timesteps):
                steeringAngleInitial = Y_train[index]
                imPath = X_train[index]
                image = cv2.imread(imPath)
                image, steeringAngle = preProcess(image, steeringAngleInitial)
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                index = index + 1
                # Make image from 66,200,3 to 3,66,200
                image_interChangedDimensions = np.transpose(image,(2,0,1))
                X_trainBatchImages[j,k,:,:,:] = image_interChangedDimensions
                Y_trainBatchImages[j,k,:] = steeringAngle

        X_trainBatchImages=np.array(X_trainBatchImages)
        Y_trainBatchImages=np.array(Y_trainBatchImages)


        yield (X_trainBatchImages,Y_trainBatchImages)


if __name__ == "__main__":

    reader=pd.read_csv('driving_log.csv')

    X_center = reader["center"]
    X_left = reader["left"].map(str.strip)
    X_right = reader["right"].map(str.strip)

    Y_center = reader["steering"]
    Y_left = reader["steering"]+0.25
    Y_right = reader["steering"]-0.25

    X_train = X_center
    Y_train = Y_center
    # X_train = np.hstack((X_center,X_left,X_right))
    # Y_train = np.hstack((Y_center,Y_left,Y_right))

    print("X_train ",len(X_train))
    print("Y_train ",len(Y_train))

    batchSize = 80
    timesteps = 10
    volumesPerBatch = int(batchSize/timesteps)

    # X,Y = batchImageGenerator(X_train,Y_train,batchSize,timesteps)
    model = buildModel(volumesPerBatch,timesteps,(3,66, 200))


    # model.fit(X_train, Y_train,batch_size=batch_size,nb_epoch=nb_epoch,validation_data=(X_validation, Y_validation),shuffle=True)
    model.fit_generator(batchImageGenerator(X_train,Y_train,batchSize=80,timesteps = 10 ),samples_per_epoch = 5000, nb_epoch=1)
    model.save_weights('model.h5')
    with open('model.json', 'w') as outfile:
        outfile.write(model.to_json())