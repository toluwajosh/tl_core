import cv2
import glob
import helper
import pickle
import os.path
import warnings
import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.externals import joblib
from distutils.version import LooseVersion
from keras.utils.np_utils import to_categorical


from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, MaxPooling2D, LSTM
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau
from keras.utils.np_utils import to_categorical

from keras import backend as K
# K.set_image_data_format('channels_first')
K.set_image_data_format('channels_last')


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def generate_arrays_from_file(data_dir, batch_size=50):
  # load samples paths
  all_files = glob.glob(data_dir+'/*.jpg')
  # sort according time created
  all_files.sort(key=os.path.getmtime)
  
  while 1:
      x=np.zeros([batch_size, 112, 112, 3])
      y=np.zeros(batch_size)
      batch_index = 0
      for i, names in enumerate(all_files):
        # print("name of file: ",names)
        # create Numpy arrays of input data
        # and labels, from each line in the file
        # extract target score
        image = cv2.imread(names)
        image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC)
        # print("Size of image: ", np.shape(image))
        label = float(names.split('-')[-1].split('.')[0])/100
        # x.append([image])
        # y.append([label])
        x[batch_index,:,:,:] = image
        y[batch_index] = label
        batch_index += 1
        # print("no ", i)
        if (i+1)%batch_size==0:
          # X = np.stack(x)
          # Y = np.stack(y)
          # print("Shape of X: ",np.shape(X))
          yield (x, y)
          x=np.zeros([batch_size, 112, 112, 3])
          y=np.zeros(batch_size)
          batch_index = 0


if __name__ == '__main__':
  


  # train properties
  epochs = 100
  batch_size = 128

  # datset
  num_classes = 1
  # dataset_dir = 'data\\processed_dataset\\all_data'
  dataset_dir = 'data/processed_dataset/all_data'

  # model parameters
  CONV = 2
  POOL = 2
  DROP = 0.3
  KERNEL = 3
  DENSE_UNITS = 1024
  INIT_ = 'he_normal'
  BORDER = 'valid'
  

  reg_val = 0.01
  pool_strides = 2

  in_w, in_h, features_used = 112, 112, 3


  # initialize model
  model = Sequential()

  # first layer
  model.add(Convolution2D(64, 
              kernel_size=(3,3),
              strides=(2,2),
              init=INIT_,
              input_shape=(in_w, in_h,features_used), 
              border_mode='same')) 
  # model.add(Dropout(DROP))
  # model.add(MaxPooling2D())
  model.add(Activation('relu'))

  # layer
  model.add(Convolution2D(96, 
              kernel_size=(3,3),
              strides=(2,2),
              init=INIT_,
              border_mode=BORDER))
  model.add(MaxPooling2D())
  model.add(Activation('relu'))
  # model.add(Dropout(DROP))

  # layer
  model.add(Convolution2D(192, 
              kernel_size=(3,3),
              strides=(2,2),
              init=INIT_,
              border_mode=BORDER))
  # model.add(MaxPooling2D())
  model.add(Activation('relu'))
  

  # layer
  model.add(Convolution2D(256, 
              kernel_size=(2,2),
              strides=(2,2),
              init=INIT_,
              border_mode=BORDER))
  # model.add(Dropout(DROP))
  # model.add(MaxPooling2D())
  model.add(Activation('relu'))

  # # layer
  # model.add(Convolution2D(128,  # you can increase this line to 128
  #             kernel_size=(2,2),
  #             strides=(1,1),
  #             init=INIT_,
  #             border_mode=BORDER))

  #%% added an LSTM layer
  model.add(Flatten())
  # model.add(Reshape((features_used,model.output_shape[1]//features_used)))
  model.add(Dense(DENSE_UNITS))
  # model.add(LSTM(DENSE_UNITS))
  model.add(Activation('relu'))
  model.add(Dropout(DROP))

  # layer
  model.add(Dense(100))
  model.add(Activation('relu'))
  model.add(Dropout(DROP))

  # last layer
  model.add(Dense(num_classes))

  model.add(Activation('softmax'))

  model.summary()

  model.compile(loss='mse',
            optimizer='adam', # adadelta
            metrics=['accuracy'])


  # training:::
  # training parameters
  model_name = 'cnn_keras'
  cross_validation_folds = 5
  EPOCHS = 200
  BATCH_SIZE = 256 # 128, 121, 192, 256 for gestures
  early_stop_patience = 3

  # %% define callbacks
  # a callback to save a list of the losses over each batch during training
  class LossHistory(Callback):
    def on_train_begin(self, logs={}):
      self.train_loss = []

    def on_batch_end(self, batch, logs={}):
      self.train_loss.append(logs.get('loss'))

  # a callback to save a list of the accuracies over each batch during training
  class AccHistory(Callback):
    def on_train_begin(self, logs={}):
      self.train_acc = []

    def on_batch_end(self, batch, logs={}):
      self.train_acc.append(logs.get('acc'))

  # callbacks
  loss_hist = LossHistory()
  acc_hist = AccHistory()
  # early_stop = EarlyStopping(monitor='val_loss', patience=early_stop_patience, verbose=0, mode='min')
  checkpoint = ModelCheckpoint('data/model_saves/keras_models/'+model_name+'_model.{epoch:02d}-{loss:.3f}', 
                                monitor='loss', verbose=0, save_best_only=True, 
                                save_weights_only=False, mode='auto')
  # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
  #           patience=3, min_lr=0.000001)

  # #%% Get Training model
  # model = get_model(features_used=features_used, 
  #                   input_shape=(x_vox, y_vox, z_vox), 
  #                   classes=CLASSES)

  print('\nBatch size:', BATCH_SIZE  )


  # model.fit_generator(train_set, y_train_in, shuffle=True, 
  #               batch_size=BATCH_SIZE, epochs=EPOCHS,
  #               verbose=1, validation_data=(test_set, y_test_in),
  #               callbacks=[loss_hist, acc_hist, early_stop, checkpoint, reduce_lr])

  model.fit_generator(generate_arrays_from_file(dataset_dir, 100), 
                      steps_per_epoch=330, epochs=EPOCHS, verbose=1,
                      callbacks=[loss_hist, acc_hist, checkpoint])

