import os
import csv
import numpy as np

train_X = np.zeros((10000,784))
train_Y = np.zeros((10000,1))
test_X = np.zeros((10000,784))
test_Y = np.zeros((10000,1))

def load_data():
    with open('../tests/digit_recog/mnist_train.csv', 'r') as file:
        reader = csv.reader(file)
        i = 0
        while i < 10000:
            row = next(reader)
            train_Y[i] = row[0]
            train_X[i] = row[1:]
            i += 1
    
    with open('../tests/digit_recog/mnist_train.csv', 'r') as file:
        reader = csv.reader(file)
        i = 0
        while i < 10000:
            row = next(reader)
            test_Y[i] = row[0]
            test_X[i] = row[1:]
            i += 1
