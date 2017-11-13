import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion


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
    self.image_input = graph.get_tensor_by_name('image_input:0')
    self.keep_prob = graph.get_tensor_by_name('keep_prob:0')
    self.output_tensor = graph.get_tensor_by_name('layer'+str(layer_out)+'_out:0')

    # # show some output
    # print('Model Properties: \n \
    #   Image Input Size: {} \
    #   Keep Probability: {} \
    #   Output Tensor Size: {}'.format(self.image_input, 
    #                                   self.keep_prob, 
    #                                   self.output_tensor))

  def layers(self, fc_layers, dropouts, mode):
    """
    :param fc_layers: list containing number of units in the fully connected layers
    :param dropouts: list containing dropout probability for each fully connected layer
    """
    assert (len(fc_layers) == len(dropouts)), \
          "The Size of fully connected layers and dropouts are not equal"
    # fc_layer = tf.layers.

    # dense = tf.reshape(self.output_tensor, [-1, 4096])

    dense = tf.layers.dense(inputs=self.output_tensor, units=fc_layers[0], 
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=dropouts[0],
                            training=mode == tf.estimator.ModeKeys.TRAIN)

    for x in range(1, len(fc_layers)-1):
      dense = tf.layers.dense(inputs=dropout, units=fc_layers[x], 
                              activation=tf.nn.relu)
      dropout = tf.layers.dropout(inputs=dense, rate=dropouts[x],
                              training=mode == tf.estimator.ModeKeys.TRAIN)

    self.logits = tf.layers.dense(inputs=dropout, units=fc_layers[-1])
    self.num_classes = fc_layers[-1]

    return self.logits

  def train(self, sess, X_train, y_train, X_validate=None, y_validate=None, 
            epochs=30, batch_size=100, keep_prob=0.4, learning_rate=None,
            save_checkpoint=None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param X_train, y_train: training data
    :param X_train, y_train: validation data
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param keep_prob: dropout keep probability
    :param learning_rate: training learning rate
    """
    label = tf.placeholder(tf.int32, shape=(None))
    self.correct_label = tf.one_hot(label, self.num_classes)

    # loss function:
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=self.correct_label, logits=self.logits))

    # set up training
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
                        self.correct_label:new_y_train[offset:end], self.keep_prob:keep_prob}) # , learning_rate:1e-4

        print("epoch: {}, batch: {}, loss: {}".format(epoch+1, current_batch, loss))
      if X_validate.any()!=None:
        print("\nEvaluating....")
        validation_accuracy = self.evaluate(X_validate, y_validate)
        print("Validation Accuracy: ", validation_accuracy)
      if save_checkpoint:
        saver.save(sess, './data/model_saves/test_train.ckpt')


  def evaluate(self, X_data, y_data, keep_prob=1.0):

    batch_size = 100

    correct_label = tf.placeholder(tf.int64, shape=(None))
    
    # correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.correct_label, 1))
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
                      correct_label:new_y_data[offset:end], self.keep_prob:keep_prob})
      total_accuracy += (accuracy * batch_size)
    return (total_accuracy / num_examples)


if __name__ == '__main__':
  data_dir = './data'

  # Load pickled data
  import pickle
  import cv2
  import numpy as np
  from keras.utils.np_utils import to_categorical
  from sklearn.utils import shuffle

  num_classes = 43

  ##
  training_file = 'data/traffic-signs-data/train.p'
  validation_file = 'data/traffic-signs-data/valid.p'
  testing_file = 'data/traffic-signs-data/test.p'

  with open(training_file, mode='rb') as f:
      train = pickle.load(f)
  with open(validation_file, mode='rb') as f:
      validate = pickle.load(f)
  with open(testing_file, mode='rb') as f:
      test = pickle.load(f)
      
  X_train, y_train = train['features'], train['labels']
  X_validate, y_validate = validate['features'], validate['labels']
  X_test, y_test = test['features'], test['labels']

  print("\nsize of training data: {}".format(np.shape(X_train)))

  y_train = to_categorical(y_train, num_classes)
  y_validate = to_categorical(y_validate, num_classes)

  print("\nsize of training labels: {}".format(np.shape(y_train)))

  num_examples = np.shape(X_train)[0]

  print("\nNumber of samples {}".format(num_examples))
  ##

  # Download pretrained vgg model
  helper.maybe_download_pretrained_vgg(data_dir)
  
  model_path = os.path.join(data_dir, 'vgg')

  model_tag = 'vgg16'
  layer_out = 7
  image_shape = (224, 224)
  epochs = 30
  batch_size = 250

  with tf.Session() as sess:
    ##
    # get_batches_fn = helper.gen_batch_function(
    #                   os.path.join(data_dir, 'data_road/training'), image_shape)
    ##

    # create transfer learning model instance
    tl_model = TL_model(sess, model_tag, model_path, layer_out)
    fc_layers = [1024, 1024, num_classes]
    dropouts = [0.4, 0.4, 1.0]

    # for training:
    mode = tf.estimator.ModeKeys.TRAIN
    # logits = tl_model.layers(fc_layers, dropouts, mode)
    tl_model.layers(fc_layers, dropouts, mode)
    tl_model.train(sess, 
                    X_train, y_train, 
                    X_validate, y_validate, 
                    batch_size=batch_size)

    # print("\nLogits: ", logits)

    # # logits = tf.reshape(logits, (-1, num_classes))
    # # correct_label = tf.reshape(label, (-1, num_classes))
    # label = tf.placeholder(tf.int32, shape=(None))
    # correct_label = tf.one_hot(label, num_classes)

    # input_image = tl_model.image_input
    # keep_prob = tl_model.keep_prob

    # print("\nInput image shape: ", input_image)


    # cross_entropy_loss = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits(
    #         labels=correct_label, logits=logits))

    # with tf.name_scope("training"):
    #   optimizer = tf.train.AdamOptimizer()
    #   # Create a variable to track the global step.
    #   global_step = tf.Variable(0, name='global_step', trainable=False)
    #   # Use the optimizer to apply the gradients that minimize the loss
    #   # (and also increment the global step counter) as a single training step.
    #   train_op = optimizer.minimize(cross_entropy_loss, global_step=global_step)


    # # to save the trained model (preparation)
    # saver = tf.train.Saver()

    # sess.run(tf.global_variables_initializer())
    # # print("\n Training... \n")
    # for epoch in range(epochs):
    #   new_X_train, new_y_train = shuffle(X_train, y_train)
    #   # Training, use batches
    #   for offset in range(0, num_examples, batch_size):
    #     end = offset + batch_size

    #     _, loss = sess.run([train_op, cross_entropy_loss], 
    #         feed_dict={input_image:new_X_train[offset:end], 
    #                     correct_label:new_y_train[offset:end], keep_prob:0.4}) # , learning_rate:1e-4

    #     print("epoch: {}, batch: {}, loss: {}".format(epoch+1, epoch, loss))
    #   saver.save(sess, './data/model_saves/test_train.ckpt')