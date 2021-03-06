import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import cv2

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

timesteps = 10
volumesPerBatch = 8
firstFrame = True
transformed_image_array = np.zeros((volumesPerBatch, timesteps, 66, 200,3), dtype="uint8")

def crop_Image(image):
    height, width = image.shape[:2]
    upper_limit = int((6.0/7.0)*height)
    lower_limit = int((2.0 / 7.0) * height)
    return image[lower_limit:upper_limit,:]

def preProcess(image):
    image = crop_Image(image)
    image= cv2.resize(image, (200, 66))
    return image

@sio.on('telemetry')
def telemetry(sid, data):
    global firstFrame
    global transformed_image_array
    global timesteps
    global volumesPerBatch
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)

    b,g,r = cv2.split(image_array)           # get b, g, r
    image_array = cv2.merge([r,g,b])
    image_array = preProcess(image_array)

    # cv2.imshow('image_orig', image_array)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # image_interChangedDimensions = np.transpose(image_array, (2, 0, 1))

    if firstFrame:
        print ("First Frame")
        #Copy the first frame to all the timesteps and volumes.
        for j in range(volumesPerBatch):
            for k in range(timesteps):
                transformed_image_array[j, k, :, :, :] = image_array
        firstFrame = False
    else:
        for j in range(volumesPerBatch):
            for k in range(timesteps):
                #Replace last frame with current frame
                if (j== (volumesPerBatch-1) and k ==(timesteps-1)):
                    transformed_image_array[j, k, :, :, :] = image_array
                elif k != (timesteps-1):
                    transformed_image_array[j, k, :, :, :] = transformed_image_array[j, k+1, :, :, :]
                else:
                    transformed_image_array[j, k, :, :, :] = transformed_image_array[j+1, 0 , :, :, :]

    # print ("transformed_image_array ", transformed_image_array.shape)
    # print ("Predicting steering angle ....")
    steering_angle = model.predict(transformed_image_array)
    # print ("steering_angle_shape",steering_angle.shape)
    steering_angle = steering_angle[volumesPerBatch-1,timesteps-1,0]
    # print("steering_angle estimated", steering_angle)
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.20
    # steering_angle = 0.001
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0.0, 0.0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)