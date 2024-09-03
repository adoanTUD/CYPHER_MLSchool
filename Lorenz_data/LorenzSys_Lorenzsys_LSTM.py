import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import layers
from numpy import linalg as LA
from tensorflow.keras.callbacks import TensorBoard
#from tensorflow.keras.layers import LSTM, Dense

import matplotlib.pyplot as plt

if os.getcwd() != '/Users/ml/Documents/Studium/Masterarbeit/Code/pemlm/LorenzSys/':
    os.chdir('/Users/ml/Documents/Studium/Masterarbeit/Code/pemlm/LorenzSys/')

saveweights = True
val_ratio = 0.8
Nstep = 50

# Read data from files
np_file = np.load('./Lorenz_data.npz')
X = np_file['X'] # Data
dt = np_file['dt']
t_split = np_file['t_stop_train']

#prepare data for training

#skip beginning
t_skip = 1
i_skip = int(t_skip/dt)
X = X[i_skip:,:]

#normalize data
avg = np.average(X,axis=0)
X=X-np.ones((X.shape[0],X.shape[1]))*avg
max = np.array([np.amax(np.absolute(X[:,0])),np.amax(np.absolute(X[:,1])),np.amax(np.absolute(X[:,2]))])
X[:,0]=X[:,0]/max[0]
X[:,1]=X[:,1]/max[1]
X[:,2]=X[:,2]/max[2]

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
for i in range(input_train.shape[0]-Nstep):
    input.append(input_train[i:i+Nstep,:])
input_train = np.array(input)
output_train = output_train[Nstep-1:-1,:]

input = []
for i in range(input_val.shape[0]-Nstep):
    input.append(input_val[i:i+Nstep,:])
input_val = np.array(input)
output_val = output_val[Nstep-1:-1,:]




#input shape (batch_size, timesteps, inputs)

input = layers.Input(shape=(Nstep,3), name='input')
lstm = layers.LSTM(1, input_shape=(Nstep,3), return_sequences=True, name='LSTM')
f_hidden = layers.Dense(units=16, activation='relu', name='f_hidden')
s_hidden = layers.Dense(units=3, activation='linear', name='output')
f_output = layers.Dense(units=3, activation='linear', name='f_output')


output = lstm(input)
output = layers.TimeDistributed(f_hidden)(output)
output = layers.TimeDistributed(s_hidden)(output)
output = layers.Flatten()(output)
output = f_output(output)

model = Model(inputs=[input], outputs=[output])

#prepare for training
model.compile(loss='mse', optimizer='adam')
model.summary()

#check if pretrained model availbale
if os.path.exists('./pretrained_1.h5'):
    print('model loads stored weights')
    model.load_weights('./pretrained_1.h5')
else:

    history = model.fit(input_train, output_train, verbose=2, validation_data=(input_val, output_val), epochs=50, batch_size=256)  # batch_size could be given default=32, also validation data can be provided
    plt.plot(history.history['loss'], '-b', label='train loss')
    plt.plot(history.history['val_loss'], '-r', label='val loss')
    plt.show()

    if saveweights and os.path.exists('./pretrained_1.h5') == False:
        print('weights saved')
        model.save_weights('./pretrained_1.h5')



# natural response

t_pred = 10
idx_end = int(t_pred/dt)
x_ref = X[idx_split:,:]
assert idx_end < x_ref.shape[0], 't_pred too long'
x_ref = x_ref[:idx_end,:]


Y = []

y_last = x_ref[0:Nstep,:].reshape((1,Nstep,3))

print('start natural response')
for i in range(x_ref.shape[0]-Nstep):
    Y.append(model.predict(y_last))
    y_last = np.append(y_last,Y[i].reshape(1,1,3),axis=1)
    y_last = y_last[:,1:,:]
    print('%d predicitions done'%(i+1))

Y = np.array(Y)
Y = Y.reshape(Y.shape[0],Y.shape[2])
Y = np.vstack((x_ref[:Nstep,:],Y))

#denormalize
Y[:,0]=Y[:,0]*max[0]
Y[:,1]=Y[:,1]*max[1]
Y[:,2]=Y[:,2]*max[2]
Y=Y+np.ones((Y.shape[0],Y.shape[1]))*avg

x_ref[:,0]=x_ref[:,0]*max[0]
x_ref[:,1]=x_ref[:,1]*max[1]
x_ref[:,2]=x_ref[:,2]*max[2]
x_ref=x_ref+np.ones((x_ref.shape[0],x_ref.shape[1]))*avg


#calculate Error
err = LA.norm(Y[Nstep:,:]-x_ref[Nstep:,:],axis=1)/(np.sum(np.square(LA.norm(x_ref[Nstep:,:],axis=1)))/x_ref[Nstep].shape[0])
plt.plot(err)
plt.show()

plt.plot(x_ref[Nstep:,0])
plt.plot(Y[Nstep:,0],'--')
plt.show()

plt.plot(x_ref[Nstep:,1])
plt.plot(Y[Nstep:,1],'--')
plt.show()

plt.plot(x_ref[Nstep:,2])
plt.plot(Y[Nstep:,2],'--')
plt.show()