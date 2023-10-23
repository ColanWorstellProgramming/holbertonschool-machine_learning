#!/usr/bin/env python3
"""
Imports
"""
# Basic libraries
import numpy as np
import pandas as pd
import random
import time
import datetime
import os

# preprocessing
from sklearn.model_selection import train_test_split

# Deeplearning
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Define mnist learning and evaluation class
class MNIST():
    def __init__(self, input_=784, output_=10,
                 layer1_out=512,
                 layer2_out=512,
                 layer1_drop=0.2,
                 layer2_drop=0.2,
                 batch_size=32,
                 epochs=10,
                 validation_split=0.1):
        self.input_ = input_
        self.output_ = output_
        self.layer1_out = layer1_out
        self.layer2_out = layer2_out
        self.layer1_drop = layer1_drop
        self.layer2_drop = layer2_drop
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.X_train, self.X_test, self.y_train, self.y_test = self.mnist_data()
        self.model = self.mnist_model()

    # Mnist dataset
    def mnist_data(self):
        # dataset
        train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
        # train dataset
        X = train.iloc[:,1:].values
        X = X.astype(np.float32)
        y = train["label"].values
        y = to_categorical(y)

        # train and valid split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        # img data
        X_train, X_test = X_train/255, X_test/255

        return X_train, X_test, y_train, y_test

    # DNN model
    def mnist_model(self):
        model = Sequential()
        model.add(Dense(self.layer1_out, input_shape=(self.input_, )))
        model.add(Activation("relu"))
        model.add(Dropout(self.layer1_drop))
        model.add(Dense(self.layer2_out))
        model.add(Activation('relu'))
        model.add(Dropout(self.layer2_drop))
        model.add(Dense(self.output_))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
        return model

    # fit mnist model
    def mnist_fit(self):
        # model file
        time_stp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        # callbacks
        es = EarlyStopping(patience=0, verbose=1)
        mc = ModelCheckpoint("mnist_model_{}.h5".format(time_stp),
                             monitor="val_loss",
                             verbose=1,
                             period=2,
                             save_best_only=True)

        # save information of model into text file
        with open("parameters_{}.txt".format(time_stp), mode="w") as f:
            f.write("-"*20)
            f.write("mnist_model : {}".format(time_stp))
            f.write("layer1_out : {}".format(self.layer1_out))
            f.write("layer1_drop : {}".format(self.layer1_drop))
            f.write("layer2_out : {}".format(self.layer2_out))
            f.write("layer2_drop : {}".format(self.layer2_drop))
            f.write("validation_split : {}".format(self.validation_split))
            f.write("batch_size : {}".format(self.batch_size))
            f.write("epochs : {}".format(self.epochs))
            f.close()

        # fit
        self.model.fit(self.X_train, self.y_train,
                         batch_size=self.batch_size,
                         epochs=self.epochs,
                         verbose=0,
                         validation_split=self.validation_split,
                         callbacks=[es, mc])
    # evaluation
    def mnist_evaluate(self):
        self.mnist_fit()

        evaluation = self.model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size, verbose=0)

        return evaluation

# run
def run_mnist(input_=784, output_=10,
              layer1_out=512,
              layer2_out=512,
              layer1_drop=0.2,
              layer2_drop=0.2,
              batch_size=16,
              epochs=16,
              validation_split=0.1):

    _mnist = MNIST(input_=input_, output_=output_,
                  layer1_out=layer1_out,
                  layer2_out=layer2_out,
                  layer1_drop=layer1_drop,
                  layer2_drop=layer2_drop,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=validation_split)

    mnist_evaluation = _mnist.mnist_evaluate()
    return mnist_evaluation

# Bayesian Optimization
# Define parmeter range
bounds = [{"name":"validation_split", "type":"continuous", "domain":(0.1, 0.3)},
          {"name":"layer1_drop", "type":"continuous", "domain":(0.0, 0.3)},
          {'name': 'layer2_drop', 'type': 'continuous', 'domain': (0.0, 0.3)},
          {'name': 'layer1_out', 'type': 'discrete', 'domain': (64, 128, 256, 512)},
          {'name': 'layer2_out', 'type': 'discrete', 'domain': (64, 128, 256, 512)},
          {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128)},
          {'name': 'epochs', 'type': 'discrete', 'domain': (10, 50, 100)}
         ]

# define function
def f(x):
    t0 = time.time()
    print(x)
    evaluation = run_mnist(layer1_drop = float(x[:,1]), # input parameter to run execution function
                           layer2_drop = float(x[:,2]),
                           layer1_out = int(x[:,3]),
                           layer2_out = int(x[:,4]),
                           batch_size = int(x[:,5]),
                           epochs = int(x[:,6]),
                           validation_split = float(x[:,0]))
    print("-"*20)
    print("loss:{0} \t accuracy:{1}".format(evaluation[0], evaluation[1]))
    print("-"*20)
    print(evaluation)
    t1 = time.time()
    print("calc time:{}".format(t1-t0))
    return evaluation[0] # only one parameter

run_mnist()

# primary explore
opt_mnist = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds) # f is optimization target function, bounds is parameter range.

opt_mnist.run_optimization(max_iter=10)

# last result
print("")
print("Optimization result")
print("optimized parameters: {}".format(opt_mnist.x_opt))
print("optimized loss: {}".format(opt_mnist.fx_opt))
