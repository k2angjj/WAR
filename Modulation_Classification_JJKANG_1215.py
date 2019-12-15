# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 00:52:00 2019

@author: jongjin.kang
"""

import os,random
os.environ["KERAS_BACKEND"] = "theano"
#os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["THEANO_FLAGS"]  = "device=cuda*"
#gpu%d"%(1)
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import random, sys, keras

import pickle
import math


with open('2016.04C.multisnr.pkl','rb') as f:
  data = pickle.load(f,encoding='bytes')


X = []
labels = [] # label each example by a pair (modulation type, snr)
total_examples = 0
for mod_type, snr in data.keys():
  current_matrix = data[(mod_type, snr)]
  total_examples += current_matrix.shape[0]
  for i in range(current_matrix.shape[0]):
    X.append(current_matrix[i])
    labels.append((str(mod_type, 'ascii'), snr)) # mod_type is of type bytes
X = np.array(X)
labels = np.array(labels)
print(f'loaded {total_examples} signal vectors into X{X.shape} and their corresponding'
      f' labels into labels{labels.shape}')



def get_fft_channel(X):
  cplx_X = 1j*X[:,1,:] + X[:,0,:]
  X_fft = np.empty_like(cplx_X).astype('float32')
  for i in range(X.shape[0]):
    X_fft[i] = np.absolute(np.fft.fft(cplx_X[0]))
        
  return X_fft.reshape(X_fft.shape[0], 1, X_fft.shape[1])

X_fft, freq_axis = get_fft_channel(X), np.fft.fftfreq(128)
print(f'computed the fft of the signals, X_fft{X_fft.shape}')


TEST_PERCENTAGE = 0.2
VALIDATION_PERCENTAGE = 0.05

EPOCHS = 20
NUM_CLASSES = 11
BATCH_SIZE = 1024

def split_data(data, labels, percentage):
  perm_idx = np.random.permutation(labels.shape[0])
  data_perm = data[perm_idx]
  labels_perm = labels[perm_idx]
  split_point = int((1-percentage)*data_perm.shape[0])
  data_train = data_perm[0:split_point]
  data_test = data_perm[split_point:]
  labels_train = labels_perm[0:split_point]
  labels_test = labels_perm[split_point:]
  y_train = labels_train[:,0]
  y_test = labels_test[:,0]
  return data_train, data_test, labels_train, labels_test, y_train, y_test


X_train, X_test, labels_train, labels_test, y_train, y_test = split_data(X,labels, TEST_PERCENTAGE)

print(f'shapes after train test split:\n'
      f'X_train{X_train.shape}, X_test{X_test.shape}, '
      f'labels_train{labels_train.shape}, labels_test{labels_test.shape}, '
      f'y_train{y_train.shape}, y_test{y_test.shape}')



def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
def generate_confusion_matrix(model, X,y, one_hot_transformer, batch_size):
    """
      y is the one hot encoded label vector passed to the model.evaluate
    """
    mod_to_idx = {mod:idx for idx,mod in enumerate(one_hot_transformer.classes_)} # use this to map modulation name to index
    y_hat = model.predict(X, batch_size)
    y_hat_onehot = np.zeros_like(y_hat)
    y_hat_onehot[np.arange(len(y_hat)), y_hat.argmax(1)] = 1 # convert the probabilities to one-hot format
    y_hat_mod = one_hot_transformer.inverse_transform(y_hat_onehot) # transform predictions to strings
    y_mod = one_hot_transformer.inverse_transform(y) # transform ground truth back to strings
    confusion_mat = np.zeros([NUM_CLASSES,NUM_CLASSES])

    acc = np.mean(y_mod == y_hat_mod)
    # fill in the confusion matrix
    for i in range(X.shape[0]):
        
        true_idx = mod_to_idx[y_mod[i]]
        pred_idx = mod_to_idx[y_hat_mod[i]]          
        confusion_mat[pred_idx,true_idx] += 1

    # normalize the matrix row wise
    for i in range(NUM_CLASSES):
        if np.sum(confusion_mat[i,:]) > 0:
          confusion_mat[i,:] /= np.sum(confusion_mat[i,:])
    
    return confusion_mat,acc




class ModelEvaluater:
  def __init__(self, model, X_train, y_train, X_test, y_test, 
               labels_test, labels, batch_size, num_partitions, model_name):
    self.model = model
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test
    self.labels = labels
    self.labels_test = labels_test
    self.model_name = model_name
    self._curr_partition = 0
    self._partition_size = X_train.shape[0]//num_partitions
    self.num_partitions = num_partitions
    self._checkpoint_callback = ModelCheckpoint(filepath='%s-weights-{epoch}.hdf5' % self.model_name,
                                                verbose=1, save_best_only=True)
    self._train_accs = []
    self._val_accs = []
    self._snr_accs = []
    
    self.batch_size = batch_size
    
    
    # get the modulation types into an array
    self.mod_types = np.unique(labels[:,0])
  
    # fit a label binarizer 
    self.mod_to_onehot = preprocessing.LabelBinarizer()
    self.mod_to_onehot.fit(self.mod_types)

    # transform the y values to one-hot encoding
    self.y_train = self.mod_to_onehot.transform(y_train)
    self.y_test = self.mod_to_onehot.transform(y_test)
    
    print(f'y_train{y_train.shape}')
    print(f'y_test{y_test.shape}')
    

    
def fit(self, epochs, val_percent=0.05, patience=4, fit_all=False, save_to_drive=False):
    curr_X, curr_y = None, None
    if fit_all:
      curr_X, curr_y = self.X_train, self.y_train
    else:
      idx_start, idx_end = self._curr_partition*self._partition_size, (self._curr_partition+1)*(self._partition_size)
      print(idx_start, idx_end, self._partition_size)
      curr_X, curr_y = self.X_train[idx_start:idx_end], self.y_train[idx_start:idx_end]
      self._curr_partition = (self._curr_partition + 1) % self.num_partitions

    callbacks = [self._checkpoint_callback, EarlyStopping(patience=patience)]
    if save_to_drive:
      callbacks.append(GDriveSaver(self.model_name))


     # train the model
    model_info = self.model.fit(curr_X, curr_y,
                                batch_size=self.batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_split=val_percent,
                                callbacks=callbacks)

    self._train_accs.extend(model_info.history['acc'])
    self._val_accs.extend(model_info.history['val_acc'])






def print_summary(self):
    # plot validation accuracy vs training accuracy
    plt.plot(np.arange(len(self._train_accs)), self._train_accs, '-o', label='training accuracy')
    plt.plot(np.arange(len(self._val_accs)), self._val_accs, '-o', label='validation accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.title(f'{self.model_name}-validation vs training accuracy')
    display(plt.show())

    # plot the confussion matrix for the whole test data
    conf_mat, avg_acc = generate_confusion_matrix(self.model, self.X_test,
                                                  self.y_test, self.mod_to_onehot, self.batch_size)
    plot_confusion_matrix(conf_mat, labels=self.mod_to_onehot.classes_,
                          title=f'{self.model_name} - conf mat for whole test data - acc={avg_acc * 100}%')
    plt.show()  
    # plot the confusion matrix per snr

    snr_accs = {}
    snrs = np.unique(np.unique(self.labels,axis=0)[:,1]).astype('int32')
    for snr in sorted(snrs):
      idx = np.where(self.labels_test[:,1]==str(snr))
      X_snr = self.X_test[idx]
      y_snr = self.y_test[idx]
      conf_mat, acc = generate_confusion_matrix(self.model, X_snr, y_snr, self.mod_to_onehot, self.batch_size)
      snr_accs[snr] = acc
      plot_confusion_matrix(conf_mat, labels=self.mod_to_onehot.classes_, 
                            title=f'{self.model_name}- Confusion Matrix (SNR={snr}) - acc={acc*100}%')
      plt.show()

    # plot the accuracy against the snr
    plt.plot(snr_accs.keys(),snr_accs.values(), '-o')
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.xticks(list(snr_accs.keys()))
    plt.show()


def build_fc_net(X1,X2):
  
  reg = 1e-3
  dropout = 0.2
  
  fc_model = Sequential()
  fc_model.add(Flatten('channels_first', input_shape=(X1,X2)))
  fc_model.add(Dense(186, activation='relu', kernel_regularizer=regularizers.l1(reg)))
  fc_model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(reg)))
  fc_model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1(reg)))
  fc_model.add(BatchNormalization())
  fc_model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l1(reg)))
  fc_model.add(Dropout(rate=dropout))
  fc_model.add(Dense(NUM_CLASSES, activation='softmax'))
  fc_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
  fc_model.summary()
  return fc_model

fc_model = build_fc_net(X.shape[1],X.shape[2])
fc_model_evaluater = ModelEvaluater(fc_model, X_train, y_train, X_test, y_test, labels_test, labels, 1024, 10, 'fc-net' )
fc_model_evaluater.fit(30,fit_all= True)


 
# =============================================================================
# 
# # Load the dataset ...
# #  You will need to seperately download or generate this file
# Xd = pickle.load(open("RML2016.10a_dict.pkl",'rb'),encoding="latin1")
# snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
# X = []  
# lbl = []
# for mod in mods:
#     for snr in snrs:
#         X.append(Xd[(mod,snr)])
#         for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
# X = np.vstack(X)
# 
# 
# 
# =============================================================================

# =============================================================================
# 
# 
# # Load the dataset ...
# 
# Xd = pickle.load(open("RML2016.10a_dict.pkl",'rb'),encoding="latin1")
# 
# # print key values of dataset
# for key in Xd:
#     print(key)
# 
# # extract Modulation signal(numpy array) from dataset  
# X_test=Xd[('QAM16', 18)]
# I_data=X_test[300,0,:]
# Q_data=X_test[300,1,:]
# 
# 
# nod=128
# Fs = 1000000                # Sampling frequency 1 MHz
# T = 1/Fs                    # Sample interval time
# te= (nod)*T                 # End of time
# t = T*np.arange(0, nod)     # Time vector
# 
# plt.figure(num=1,dpi=50,facecolor='white')
# plt.subplot(2,1,1)
# plt.plot(t,I_data,'b')
# plt.grid()
# #plt.xlim( 0, 0.05)
# plt.subplot(2,1,2)
# plt.plot(t,Q_data,'r')
# plt.xlabel('time($sec$)')
# plt.ylabel('y')
# plt.grid()
# plt.savefig("./test_figure1.png",dpi=300)
# 
# 
# 
# 
# 
# 
# 
# C_data=I_data+Q_data*1j     # make complex data for complex fft
# 
# # Calculate FFT ....................
# n=nod                                    # Length of signal
# NFFT=n                                   # ?? NFFT=2^nextpow2(length(y))  ??
# k=np.arange(NFFT)
# f0=k*Fs/NFFT                             # double sides frequency range
# #f0=np.fft.fftshift(f0)
# #f0=f0[range(math.trunc(NFFT/2))]        # single sied frequency range
# 
# Y=np.fft.fft(C_data)/NFFT                # fft computing and normaliation
# Y=np.fft.fftshift(Y)
# #Y=Y[range(math.trunc(NFFT/2))]          # single sied frequency range
# amplitude_Hz = 2*abs(Y)
# phase_ang = np.angle(Y)*180/np.pi
# 
# # Plot amplitude spectrum.
# plt.figure(num=2,dpi=50,facecolor='white')
# plt.subplot(2,1,1)
# plt.plot(f0,amplitude_Hz,'b')   
# plt.title('Fast Fourier Transform')
# plt.xlabel('frequency($Hz$)')
# plt.ylabel('amplitude')
# plt.grid()
# 
# # Phase ....
# plt.subplot(2,1,2)
# plt.plot(f0,phase_ang,'r')   
# plt.xlabel('frequency($Hz$)')
# plt.ylabel('phase($deg.$)')
# plt.grid()
# 
# plt.savefig("./test_figure2.png",dpi=300)
# 
# =============================================================================
