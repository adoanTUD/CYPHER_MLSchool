import os
import numpy as np
import tensorflow as tf
import argparse

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from numpy import linalg as LA
from tensorflow.keras.callbacks import TensorBoard
#from tensorflow.keras.layers import LSTM, Dense

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-u','--units',help='Number of hidden LSTM units', type=int)
parser.add_argument('-l','--lookback',help='Number of previous timesteps for prediciton', type=int)
parser.add_argument('-e','--epochs',help='Number of training epochs', type=int)
args = parser.parse_args()

saveweights = True
lookback = args.lookback #10
epsilon = 0.2
statistics = False
normalization = 1
attractor = False

# Read data from files
os.chdir("/Users/ml/Documents/Studium/Masterarbeit/Code/pemlm/LorenzSys/")
np_file = np.load('./Lorenz_data.npz')
X = np_file['X'] # Data
dt = np_file['dt']
t_split = np_file['t_stop_train']
t_skip = np_file['t_skip']
val_ratio = np_file['val_ratio']
Lyap = np_file['Lyap']

#prepare data for training

#skip beginning
i_skip = int(t_skip/dt)
X = X[i_skip:,:]

#normalize data
stddev = np.std(X,axis=0)
avg = np.average(X,axis=0)

if normalization == 0:
    X=(X-avg)/stddev

if normalization == 1:
    X = X - avg
    max = np.array([np.amax(np.absolute(X[:, 0])), np.amax(np.absolute(X[:, 1])), np.amax(np.absolute(X[:, 2]))])
    X = X / max


input_all = X[:-1,:]
output_all = X[1:,:]

#split data into training, validation
idx_split = int(t_split/dt) - i_skip
assert idx_split > 0 , 'skip time is bigger than split time'
#index that seperates training and validation data
idx_val = int(idx_split * (1 - val_ratio))

input_train_val = input_all[:idx_split,:]
output_train_val = output_all[:idx_split,:]

input_train = input_train_val[:-idx_val,:]
output_train = output_train_val[:-idx_val,:]

input_val = input_train_val[input_train_val.shape[0]-idx_val:,:]
output_val = output_train_val[output_train_val.shape[0]-idx_val:,:]

#reshape the data
input = []
for i in range(input_train.shape[0] - lookback):
    input.append(input_train[i:i + lookback, :])
input_train = np.array(input)
output_train = output_train[lookback - 1:-1, :]

input = []
for i in range(input_val.shape[0] - lookback):
    input.append(input_val[i:i + lookback, :])
input_val = np.array(input)
output_val = output_val[lookback - 1:-1, :]




#input shape (batch_size, timesteps, inputs)


model = Sequential()
model.add(LSTM(args.units, input_shape=(lookback, 3))) #5
model.add(Dense(3,activation=None))
#model.add(Dense(3,activation='linear'))

#prepare for training
model.compile(loss='mse', optimizer='adam')
model.summary()

#check if pretrained model availbale
if os.path.exists('./pretrained.h5'):
    print('model loads stored weights')
    model.load_weights('./pretrained.h5')
else:

    history = model.fit(input_train, output_train, verbose=2, validation_data=(input_val, output_val), epochs=args.epochs)  # batch_size could be given default=32, also validation data can be provided
    plt.plot(history.history['loss'], '-b', label='train loss')
    plt.plot(history.history['val_loss'], '-r', label='val loss')
    plt.show()

    if saveweights and os.path.exists('./pretrained.h5') == False:
        print('weights saved')
        model.save_weights('./pretrained.h5')



if statistics == True:

    file = open("prediction.txt","a")
    file.write("units=" + str(args.units) + ", lookback=" + str(args.lookback) + ", epochs=" + str(args.epochs) + "\n")

    # natural response
    x_ref = X[idx_split:, :]  # whole test set
    t_pred = 10 / Lyap
    err_t = []

    for i in range(100):
        idx_start = int(i * t_pred / dt)
        idx_end = int((i + 1) * t_pred / dt)

        assert idx_end < x_ref.shape[0], 't_pred too long'
        x_loc = x_ref[idx_start:idx_end, :]  # local reference solution

        Y = []
        y_last = x_loc[0:lookback, :].reshape((1, lookback, 3))

        for j in range(x_loc.shape[0] - lookback):
            Y.append(model.predict(y_last))
            y_last = np.append(y_last, Y[j].reshape(1, 1, 3), axis=1)
            y_last = y_last[:, 1:, :]
            print('Run %d, %d predicitions done' % (i, (j + 1)))

        Y = np.array(Y)
        Y = Y.reshape(Y.shape[0], Y.shape[2])
        Y = np.vstack((x_loc[:lookback, :], Y))

        # denormalize
        if normalization == 0:
            Y = Y * stddev + avg
            x_loc = x_loc * stddev + avg
        
        if normalization == 1:
            Y = Y * max + avg
            x_loc = x_loc * max + avg

        # calculate Error
        err = LA.norm(Y[lookback:, :] - x_loc[lookback:, :], axis=1) / np.sqrt(
            np.average(np.square(LA.norm(x_loc[lookback:, :], axis=1))))
        t = (np.argmax(err > epsilon) + 1) * dt

        err_t.append(t)

    err_t = np.array(err_t)
    print('mean:%f' % np.average(err_t))
    print('standarddeviation:%f' % np.std(err_t))

    file.write("mean:" + str(np.average(err_t)) + "\n")
    file.write("stddev:" + str(np.std(err_t)) + "\n")
    file.close()

else:

    t_pred = 10 / Lyap
    idx_end = int(t_pred / dt)
    x_ref = X[idx_split:, :]
    assert idx_end < x_ref.shape[0], 't_pred too long'
    x_ref = x_ref[:idx_end, :]

    Y = []

    y_last = x_ref[0:lookback, :].reshape((1, lookback, 3))

    print('start natural response')
    for i in range(x_ref.shape[0] - lookback):
        Y.append(model.predict(y_last))
        y_last = np.append(y_last, Y[i].reshape(1, 1, 3), axis=1)
        y_last = y_last[:, 1:, :]
        print('%d predicitions done' % (i + 1))

    Y = np.array(Y)
    Y = Y.reshape(Y.shape[0], Y.shape[2])
    Y = np.vstack((x_ref[:lookback, :], Y))

    # denormalize
    if normalization == 0:
        Y = Y * stddev + avg
        x_ref = x_ref * stddev + avg

    if normalization == 1:
        Y = Y * max + avg
        x_ref = x_ref * max + avg

    # calculate Error
    err = LA.norm(Y[lookback:, :] - x_ref[lookback:, :], axis=1) / np.sqrt(
        np.average(np.square(LA.norm(x_ref[lookback:, :], axis=1))))
    plt.plot(err)
    plt.show()

    plt.plot(x_ref[lookback:, 0])
    plt.plot(Y[lookback:, 0], '--')
    plt.show()

    plt.plot(x_ref[lookback:, 1])
    plt.plot(Y[lookback:, 1], '--')
    plt.show()

    plt.plot(x_ref[lookback:, 2])
    plt.plot(Y[lookback:, 2], '--')
    plt.show()

if attractor == True:
    t_pred = 150 / Lyap

    x_ref = X[idx_split:, :]
    Y = []
    y_last = x_ref[0:lookback, :].reshape((1, lookback, 3))

    print('start natural response for Attractor')
    for i in range(int(t_pred / dt)):
        Y.append(model.predict(y_last))
        y_last = np.append(y_last, Y[i].reshape(1, 1, 3), axis=1)
        y_last = y_last[:, 1:, :]
        print('%d predicitions done' % (i + 1))

    Y = np.array(Y)
    Y = Y.reshape(Y.shape[0], Y.shape[2])
    Y = np.vstack((x_ref[:lookback, :], Y))

    # denormalize
    if normalization == 0:
        Y = Y * stddev + avg

    if normalization == 1:
        Y = Y * max + avg

    plt.plot(Y[:, 0], Y[:, 1], linewidth=0.1)
    plt.show()

    plt.plot(Y[:, 0])
    plt.show()

    plt.plot(Y[:, 1])
    plt.show()
