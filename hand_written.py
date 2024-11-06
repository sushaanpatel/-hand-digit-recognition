import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from autils import *
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from data import train_X, train_Y, test_X, test_Y, load_data

load_data()

model = Sequential([
    tf.keras.layers.InputLayer(shape=(784,)),
    Dense(units=100, activation='relu'),
    Dense(units=45, activation='relu'),
    Dense(units=35, activation='relu'),
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=10, activation='linear')
])

layer_num = 6
lock_weight = True
# # * setting weights
for i in range(1, layer_num+1):
    weight = np.loadtxt(f'params/w{i}.csv', delimiter=',')
    bias = np.loadtxt(f'params/b{i}.csv', delimiter=',')
    model.layers[i-1].set_weights([weight,bias])

model.summary()

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

# * saving weights
if not lock_weight:
    model.compile(loss = SparseCategoricalCrossentropy(from_logits=True), optimizer=Adam(0.0001))

    model.fit(train_X, train_Y, epochs=20, batch_size=32)
    
    for i in range(1, layer_num+1):
        W,b = model.layers[i-1].get_weights()
        np.savetxt(f'params/w{i}.csv', W, delimiter=',')
        np.savetxt(f'params/b{i}.csv', b, delimiter=',')

# * output handling
true = 0
test_size = 100
temp = model.predict(test_X)
out = tf.nn.softmax(temp).numpy()
for i in range(test_size):
    if test_Y[i] == np.argmax(out[i]):
        print( f"{test_Y[i]}, category: {np.argmax(out[i])}")
        true += 1
    else:
        print( f"{test_Y[i]}, category: {np.argmax(out[i])},        FALSE")

print("\nAccuracy: ", true/test_size)
print("Error Rate: ", 1 - (true/test_size), f", {int((1 - (true/test_size))*test_size)} wrong")