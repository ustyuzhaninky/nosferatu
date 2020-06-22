#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import datetime
import argparse
import re
import glob
from nosferatu.neocortex.Online import OnlineBolzmannCell
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
from skimage.transform import resize
from tensorflow.keras import metrics
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix
from tensorflow.keras.wrappers.scikit_learn from tensorflow import kerasClassifier
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import categorical_accuracy

tf.config.experimental_run_functions_eagerly(True)

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

tf.config.experimental_run_functions_eagerly(True)

def trainRBM(data, learning_rate, k1, k2, epochs, batch_size, dims):
    # import data
    print("importing training data")
    if data == "fashion_mnist":
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif data == "mnist":
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
    elif data == "faces":
        x_train = [resize(mpimg.imread(file),(28,28)) for file in glob.glob("data/faces/*")]
        x_train = np.asarray(x_train)
        # make images sparse for easier distinctions
        for img in x_train:
            img[img < np.mean(img)+0.5*np.std(img)] = 0
    else:
        raise NameError("unknown data type: %s" % data)
    if data == "mnist" or data == "fashion_mnist":
        x_train = x_train/255.0
        x_test = x_test/255.0
        x_train = tf.cast(tf.reshape(x_train, shape=(x_train.shape[0], 784)), "float32")
        x_test = tf.cast(tf.reshape(
            x_test, shape=(x_test.shape[0], 784)), "float32")
    elif data == "faces":
        # auto conversion to probabilities in earlier step
        x_train = tf.cast(tf.reshape(
            x_train, shape=(x_train.shape[0], 784)), "float32")

    # train_path = tf.keras.utils.get_file(
    #     "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
    # test_path = tf.keras.utils.get_file(
    #     "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
    # CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
    #                     'PetalLength', 'PetalWidth', 'Species']
    # # SPECIES = ['Setosa', 'Versicolor', 'Virginica']
    # train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    # test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

    # y_train, y_test = train.pop('Species').values, test.pop('Species').values
    # x_train, x_test = train.values, test.values
    # y_train = tf.keras.utils.to_categorical(y_train, 3)
    # y_test = tf.keras.utils.to_categorical(y_test, 3)

    feature_dim = x_train.shape[-1]
    output_dim = y_train.shape[-1]

    # x_train = x_train.reshape(x_train.shape[0], x_train.shape[-1])
    # create log directory
    # parse string input into integer list
    inputs = tf.keras.Input(feature_dim)
    outputs = OnlineBolzmannCell(50, online=True, return_hidden=True)(inputs)
    outputs = tf.keras.layers.Dropout(0.2)(outputs)
    # outputs = tf.keras.layers.Dense(100)(inputs)
    outputs = tf.keras.layers.Dense(output_dim, activation='sigmoid')(outputs)
    rbm = tf.keras.Model(inputs, outputs)
    rbm.compile(loss='categorical_crossentropy',
                optimizer="adam", metrics=["accuracy"])
    print('Training...')
    rbm.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_split=0.1)
    for i in range(epochs):
        pred = rbm.predict(x_train)
        score = rbm.evaluate(x_train, y_train, verbose=0)
        print(f'Epoch {i+1}, loss={score[0]} Accuracy={score[1]}')
    score = rbm.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print(10*'#')
    score_train = classification_report(
        y_train.astype(np.bool), rbm.predict(x_train).astype(np.bool))
    score_test = classification_report(
        y_test, rbm.predict(x_test).astype(np.bool))
    acc_train = categorical_accuracy(
        y_train, rbm.predict(x_train))
    acc_test = categorical_accuracy(
        y_test, rbm.predict(x_test))
    print(f'Train score:\nAccuracy: {acc_train}\n{score_train}')
    print(f'Test score:\nAccuracy: {acc_test}\n{score_test}')
    print(10*'#')
    # plot_confusion_matrix(KerasClassifier(rbm, batch_size=batch_size), x_train, y_train>0.51)
    # plot_confusion_matrix(KerasClassifier(
        # rbm, batch_size=batch_size), x_test, y_test > 0.51)
    plt.show()

def getCurrentTime():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

####################################
# main command call
####################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="mnist",
            help="data source to train RBM, possibilities are 'mnist', 'fashion_mnist' and 'faces' <default: 'mnist'>")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="learning rate for stacked RBMs <default: 0.01>")
    parser.add_argument("--k1", type=int, default=1,
            help="number of Gibbs-sampling steps pre-PCD-k algorithm <default: 1>")
    parser.add_argument("--k2", type=int, default=5,
            help="number of Gibbs-sampling steps during PCD-k algorithm <default: 5>")
    parser.add_argument("--epochs", type=int, default=1,
            help="number of overall training data passes for each RBM <default: 1>")
    parser.add_argument("--batch-size", type=int, default=32,
            help="size of training data batches <default: 32>")
    # requiredNamed = parser.add_argument_group('required named arguments')
    # requiredNamed.add_argument('-d', '--dimensions', type=str, 
                            #    help="consecutive enumeration of visible and hidden units separated by a comma character, eg. 784,500", 
                            #    required=True)
    args = parser.parse_args()
    # train RBM based on parameters
    trainRBM(args.data,args.learning_rate,args.k1,args.k2,args.epochs,args.batch_size,None)
