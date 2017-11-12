import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion


class TL_model(object):
  """docstring for TL_model"""
  def __init__(self, model_tag, model_path, layer_out):
    super(TL_model, self).__init__()
    """
    :param model_path: path to model to use for transfer learning
    :param layer_out: integer value of the cutoff layer for the transfer learning model
    """
    self.model_tag = model_tag
    self.model_path = model_path

    # load model from path
    tf.save_model.loader.load(sess, [self.model_tag], self.model_path)

    # model properties
    self.image_input = graph.get_tensor_by_name('image_input:0')
    self.keep_prob = graph.get_tensor_by_name('keep_prob:0')
    self.output_tensor = graph.get_tensor_by_name('layer'+str(layer_out)+'_out:0')

    # show some ouput
    print('Model Properties: \n \
      Image Input Size: {} \
      Keep Probability: {} \
      Output Tensor Size: {}'.format(self.image_input, 
                                      self.keep_prob, 
                                      self.output_tensor))

  def model_update(self):
    print('no implemented yet')
    pass

if __name__ == '__main__':
  data_dir = './data'
  
  # Download pretrained vgg model
  helper.maybe_download_pretrained_vgg(data_dir)
  
  model_path = os.path.join(data_dir, 'vgg')

  model_tag = 'vgg16'
  layer_out = 7

  # create transfer learning model instance
  tl_model = TL_model(model_tag, model_path, layer_out)