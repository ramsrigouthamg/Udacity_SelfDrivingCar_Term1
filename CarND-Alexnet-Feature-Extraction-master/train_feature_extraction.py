#!/usr/bin/python
# -*- coding: utf-8 -*-
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import time
from sklearn.utils import shuffle

# TODO: Load traffic signs data.

training_file = 'train.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

# TODO: Split data into training and validation sets.
nb_classes = 43
rate = 0.001
(X_train, y_train) = (train['features'], train['labels'])
(X_test, y_test) = (test['features'], test['labels'])
(X_train, y_train) = shuffle(X_train, y_train)
(X_train, X_validation, y_train, y_validation) = \
    train_test_split(X_train, y_train, test_size=0.1, random_state=0,
                     stratify=y_train)

# TODO: Define placeholders and resize operation.

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int64, None)
resize_images = tf.image.resize_images(x, (227, 227))
# with tf.device('/cpu:0'):
#     one_hot_y = tf.one_hot(y, 43)

# TODO: pass placeholder as first argument to `AlexNet`.

fc7 = AlexNet(resize_images, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!

fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
        y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation,var_list=[fc8W, fc8b])
correct_prediction = tf.arg_max(logits, 1)
accuracy_operation = tf.reduce_mean(tf.cast(tf.equal(correct_prediction, y), tf.float32))

# TODO: Train and evaluate the feature extraction model.
# Training the model.

EPOCHS = 10
BATCH_SIZE = 256
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        (batch_x, batch_y) = (X_data[offset:offset + BATCH_SIZE],
                              y_data[offset:offset + BATCH_SIZE])
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x,
                            y: batch_y})
        total_accuracy += accuracy * len(batch_x)
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    num_examples = len(X_train)
    print ('Training...')
    print ()
    for i in range(EPOCHS):
        t = time.time()
        print ("EPOCH {} ...".format(i+1))
        (X_train, y_train) = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            (batch_x, batch_y) = (X_train[offset:end],
                                  y_train[offset:end])
            sess.run(training_operation, feed_dict={x: batch_x,
                     y: batch_y})

        print ("Time: %.3f seconds" % (time.time() - t))
        validation_accuracy = evaluate(X_validation, y_validation)

        print ("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print ()

    saver.save(sess, '.\FeatureExtractionResults')
    print ('Model saved')

# Testing the model

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test, y_test)
    print ("Test Accuracy = {:.3f}".format(test_accuracy))
