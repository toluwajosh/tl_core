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


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


class TF_data(object):
  """docstring for TF_data"""
  def __init__(self, batch_size=None):
    super(TF_data, self).__init__()
    if batch_size:
      self.batch_size = batch_size


  def make_data(self, data_dir):
    """
    make a tensorflow dataset
    """
    samples = []
    labels = []

    # load samples paths
    all_files = glob.glob(data_dir+'/*.jpg')

    # sort according time created
    all_files.sort(key=os.path.getmtime)
    for i, names in enumerate(all_files):
  
      # extract target score
      label = float(names.split('-')[-1].split('.')[0])/100

      # append data:
      samples.append(names)
      labels.append(label)

    self.num_of_samples = len(samples)

    data = tf.data.Dataset.from_tensor_slices((samples, labels))
    data = data.map(self._input_parser)
    
    # use batch size
    if self.batch_size:
      data = data.batch(self.batch_size)

    return data



  def _input_parser(self, img_path, label):
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_jpeg(img_file, channels=3)

    # do some preprocessing: reshape, etc..
    # img_decoded = tf.image.resize_images(img_decoded, [224,224])
    img_decoded = tf.image.resize_images(img_decoded, [112,112])

    return img_decoded, label


  def make_iterator(self, data_dir):
    self.data = self.make_data(data_dir)
    
    self.iterator = self.data.make_initializable_iterator()
    # self.iterator = tf.data.Iterator.from_structure(self.data.output_types, self.data.output_shapes)


  def initialize_iterator(self):
    # data = self.make_data(data_dir)
    self.iterator.make_initializer(self.data).initializer
    print("Initialized!!!")



class TL_model(object):
  """docstring for TL_model"""
  def __init__(self, sess, model_tag, model_path, layer_out):
    super(TL_model, self).__init__()
    """
    :param model_path: path to model to use for transfer learning
    :param layer_out: integer value of the cutoff layer for the transfer learning model
    """
    self.model_tag = model_tag
    self.model_path = model_path

    # load model from path
    tf.saved_model.loader.load(sess, [self.model_tag], self.model_path)

    # model properties
    graph = tf.get_default_graph()

    with tf.name_scope('input'):
      self.image_input = graph.get_tensor_by_name('image_input:0')
    
    self.keep_prob = graph.get_tensor_by_name('keep_prob:0')
    self.output_tensor = graph.get_tensor_by_name('layer'+str(layer_out)+'_out:0')

    # define a placeholder to control dropout
    self.network_mode = tf.placeholder(tf.bool)

  def fc_layers(self, fc_units, dropouts):
    """
    :param fc_units: list containing number of units in the fully connected layers
    :param dropouts: list containing dropout probability for each fully connected layer
    """
    

    assert (len(fc_units) == len(dropouts)), \
          "The Size of fully connected layers and dropouts are not equal"

    self.output_tensor = tf.reshape(self.output_tensor, [-1, self.output_tensor.get_shape()[-1]])

    dense = tf.layers.dense(inputs=self.output_tensor, units=fc_units[0], 
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=dropouts[0],
                            training=self.network_mode == tf.estimator.ModeKeys.TRAIN)

    for x in range(1, len(fc_units)-1):
      dense = tf.layers.dense(inputs=dropout, units=fc_units[x], 
                              activation=tf.nn.relu)
      dropout = tf.layers.dropout(inputs=dense, rate=dropouts[x],
                              training=self.network_mode == tf.estimator.ModeKeys.TRAIN)

    self.logits = tf.layers.dense(inputs=dropout, units=fc_units[-1])
    self.num_classes = fc_units[-1]


  def rc_layers(self, rec_units, dropouts):

    self.num_classes = rec_units[-1]
    num_hidden = rec_units[0]

    # input data to lstm cell should be in the format: [batch_size, sequence_length, input_dimension]
    # -1 means, to infer the size of the input
    rc_input = tf.reshape(self.output_tensor, [-1, 1, self.output_tensor.get_shape()[-1]])

    cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)

    outputs, _ = tf.nn.dynamic_rnn(cell, rc_input, dtype=tf.float32)

    outputs = tf.reshape(outputs, [-1, num_hidden])

    weight = tf.get_variable("weight", [num_hidden, self.num_classes])
    bias = tf.get_variable("bias", [self.num_classes])

    self.logits = tf.matmul(outputs, weight) + bias


  def multi_rc_layers(self, rec_units, dropouts):

    self.num_classes = rec_units[-1]
    num_hidden = rec_units[-2]

    # print("\nVGG Output Tensor: ", self.output_tensor)

    # input data to lstm cell should be in the format: [batch_size, sequence_length, input_dimension]
    # -1 means, to infer the size of the input
    rc_input = tf.reshape(self.output_tensor, [-1, 1, self.output_tensor.get_shape()[-1]])
    # print("\nrc input Tensor: ", rc_input)

    # In case of multiple layers: here 2 LSTMCells
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in rec_units[:-1]]
    # Then: create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers, state_is_tuple=True)

    # defining initial state, 
    # TODO: Implement initial_state. Need to specify a batch size
    # right now batch size is not a global variable or placeholder
    # initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell, 
                                    inputs=rc_input, 
                                    # initial_state=initial_state,
                                    dtype=tf.float32)

    outputs = tf.reshape(outputs, [-1, num_hidden])

    print("\n>> LSTM output: ",outputs)

    weight = tf.get_variable("weight", [num_hidden, self.num_classes])
    bias = tf.get_variable("bias", [self.num_classes])
  
    self.logits = tf.matmul(outputs, weight) + bias


  def train(self, sess, X_train, y_train, X_validate=None, y_validate=None, 
            epochs=30, batch_size=100, keep_prob=0.4, learning_rate=None,
            save_checkpoint=None):
    """
    Train neural network and print out the loss during training.
    
    Args:
      sess: TF Session
      X_train, y_train: training data
      X_train, y_train: validation data
      epochs: Number of epochs
      batch_size: Batch size
      keep_prob: dropout keep probability
      learning_rate: training learning rate
    
    Returns:
    """
    with tf.name_scope('input'):
      label = tf.placeholder(tf.int32, shape=(None))
    correct_label = tf.one_hot(label, self.num_classes)

    # loss function:
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=correct_label, logits=self.logits))

    # set up training
    # first set up optimizer
    with tf.name_scope("training"):
      optimizer = tf.train.AdamOptimizer()
      # Create a variable to track the global step.
      global_step = tf.Variable(0, name='global_step', trainable=False)
      # Use the optimizer to apply the gradients that minimize the loss
      # (and also increment the global step counter) as a single training step.
      train_op = optimizer.minimize(cross_entropy_loss, global_step=global_step)

    if save_checkpoint:
      # to save the trained model (preparation)
      saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    # print("\n Training... \n")
    for epoch in range(epochs):
      new_X_train, new_y_train = shuffle(X_train, y_train)
      
      # Training, use batches
      current_batch = 0
      for offset in range(0, num_examples, batch_size):
        current_batch += 1
        end = offset + batch_size

        _, loss = sess.run([train_op, cross_entropy_loss], 
            feed_dict={self.image_input:new_X_train[offset:end], 
                        correct_label:new_y_train[offset:end], 
                        self.keep_prob:keep_prob,
                        self.network_mode:tf.estimator.ModeKeys.EVAL}) # , learning_rate:1e-4

        print("epoch: {}, batch: {}, loss: {}".format(epoch+1, current_batch, loss))
      if X_validate.any()!=None:
        print("\nEvaluating....")
        validation_accuracy = self.evaluate(X_validate, y_validate)
        print("Validation Accuracy: ", validation_accuracy)
      if save_checkpoint:
        saver.save(sess, save_checkpoint)

  def train_on_iter(self, sess, tf_data, data_dir, validation_data=None, 
            epochs=30, batch_size=100, keep_prob=0.4, learning_rate=None,
            save_checkpoint=None):

    prediction = tf.sigmoid(self.logits)
    
    with tf.name_scope('input'):
      label = tf.placeholder(tf.float32, shape=(None), name='label')

    # print("Number of samples: ", train_data.num_of_samples)

    # loss function:
    with tf.name_scope('train_loss'):
      train_loss = tf.reduce_mean(
        tf.losses.mean_squared_error(
            labels=label, predictions=prediction))

    # set up training
    # first set up optimizer
    with tf.name_scope("training"):
      optimizer = tf.train.AdamOptimizer()
      # Create a variable to track the global step.
      global_step = tf.Variable(0, name='global_step', trainable=False)
      # Use the optimizer to apply the gradients that minimize the loss
      # (and also increment the global step counter) as a single training step.
      train_op = optimizer.minimize(train_loss, global_step=global_step)


    if self.load_prev_model(save_checkpoint):
      print("Loaded previous model...")
      saver = tf.train.Saver()
    elif save_checkpoint:
      # to save the trained model (preparation)
      saver = tf.train.Saver()
      sess.run(tf.global_variables_initializer())
    else:
      sess.run(tf.global_variables_initializer())

    # for tensorboard
    writer = tf.summary.FileWriter('data/tb/', graph=tf.get_default_graph())

    # print("\n Training... \n")
    for epoch in range(epochs):
      tf_data.make_iterator(data_dir)
      
      # initialize iterator
      # tf_data.initialize_iterator()
      sess.run(tf_data.iterator.initializer)

      next_element = tf_data.iterator.get_next()

      # Training, use batches
      # initialize iterator
      # sess.run(train_data.iterator().initializer)
      # 0, train_data.num_of_samples, batch_size
      current_batch = 0
      # for batch in range(5):
      for batch in range(0, tf_data.num_of_samples, batch_size):
        current_batch += 1

        try:
          X_train = sess.run(next_element)[0]
          y_train = sess.run(next_element)[1]
        except tf.errors.OutOfRangeError:
          print("End of dataset")
          break

        _, loss = sess.run([train_op, train_loss], 
            feed_dict={self.image_input:X_train, 
                        label:y_train, 
                        self.keep_prob:keep_prob,
                        self.network_mode:tf.estimator.ModeKeys.EVAL}) # , learning_rate:1e-4

        print("epoch: {}, batch: {}, loss: {}".format(epoch+1, current_batch, loss))



      # # if X_validate.any()!=None:
      # #   print("\nEvaluating....")
      # #   validation_accuracy = self.evaluate(X_validate, y_validate)
      # #   print("Validation Accuracy: ", validation_accuracy)
      if save_checkpoint:
        print("Saving checkpoint...")
        saver.save(sess, save_checkpoint)


  def evaluate(self, X_data, y_data, keep_prob=1.0):

    batch_size = 100

    correct_label = tf.placeholder(tf.int64, shape=(None))
    
    correct_prediction = tf.equal(tf.argmax(self.logits, 1), correct_label)
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    num_examples = np.shape(X_data)[0]
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
      end = offset + batch_size
      new_X_data, new_y_data = shuffle(X_data, y_data)
      accuracy = sess.run(accuracy_operation, 
        feed_dict={self.image_input:new_X_data[offset:end], 
                      correct_label:new_y_data[offset:end], 
                      self.keep_prob:keep_prob,
                      self.network_mode:tf.estimator.ModeKeys.EVAL})
      total_accuracy += (accuracy * batch_size)
    return (total_accuracy / num_examples)

  def test(self, X_data, y_data, model_save_path=None, keep_prob=1.0):
    if self.load_prev_model(model_save_path):
      accuracy = self.evaluate(X_data, y_data)
      print("Test Accuracy: ", accuracy)
    else:
      exit()

  def load_prev_model(self, model_save_path):
    try:
      meta_graph_file = model_save_path+'.meta'
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("meta_graph_file: ",meta_graph_file)
        new_saver = tf.train.import_meta_graph(meta_graph_file)
        new_saver.restore(sess, model_save_path)
      return 1
    except Exception as e:
      print("\nNo previous model found, or saved model specified!")
      return 0


if __name__ == '__main__':
  
  # train properties
  epochs = 100
  batch_size = 100


  pretrained_model_dir = './data'
  # Download pretrained vgg model, if not downloaded yet
  helper.maybe_download_pretrained_vgg(pretrained_model_dir)
  
  # load pretrained model
  model_path = os.path.join(pretrained_model_dir, 'vgg')
  model_tag = 'vgg16'
  layer_out = 7
  

  num_classes = 1
  dataset_dir = 'data/processed_dataset/all_data'

  tf_data = TF_data(batch_size)



  with tf.Session() as sess:

    # create transfer learning model instance
    tl_model = TL_model(sess, model_tag, model_path, layer_out)

    # define added layers
    # fc_units = [1024, 1024, num_classes]
    # rc_units = [128, 256, num_classes]
    rc_units = [256, 128, 64, num_classes]
    dropouts = [0.4, 0.4, 1.0]

    # for training:
    # tl_model.fc_layers(fc_units, dropouts) # for fully connected layers
    tl_model.multi_rc_layers(rc_units, dropouts) # for lstm layers

    print("\nLogits: ", tl_model.logits)
    
    tl_model.train_on_iter(sess, 
                    tf_data,
                    dataset_dir, 
                    epochs=epochs,
                    batch_size=batch_size,
                    save_checkpoint='./data/model_saves/rc_train_2.ckpt')


    # # tl_model.train(sess, 
    # #                 X_train, y_train, 
    # #                 X_validate, y_validate,
    # #                 epochs=epochs,
    # #                 batch_size=batch_size,
    # #                 save_checkpoint='./data/model_saves/rc_test_train.ckpt')


    # # tl_model.test(X_test, y_test, 
    # #                 model_save_path='./data/model_saves/test_train.ckpt')