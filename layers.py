from tensorflow.python.framework import ops
import os
import warnings
import tensorflow as tf

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()


def new_weights(shape):
    return tf.Variable(tf.compat.v1.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


class Input:
  def __init__(self , input_shape):
    """
    input_shape : the input for the model
    """
    if len(input_shape)!= 3:
      raise ValueError ("The shape Must be 3 dimention ")
    self.img_size = input_shape 

  def apply(self):
       return tf.compat.v1.placeholder(tf.float32, shape=[None, self.img_size[0] , self.img_size[1] , self.img_size[2]], name='x')

class Conv2D:
  def __init__(self , kernel_size , num_filters , strides = 1 , padding = "VAID" , activation = "relu"  ):
    """
    constractor to Apply Conv method 
    kernel_size : tuple or int of numbers
    num_filters : numbers of filter on this layer
    strides     : strides wich want to apply
    padding     : padding of the image 
    activation  : activation function
    """
    # verifed number of kernel_size 
    if ( type(kernel_size ) == int ):
      self.kernel_size = kernel_size
    elif (type(kernel_size) in [tuple , list] ):
      self.kernel_size = kernel_size[0] 
    else:
      raise ValueError("type of the kernel_size must be int , tuple or list  ")

    # verifed of number of filters
    if(type(num_filters) != int):
      raise ValueError(" numbers of filters must be int ")
    self.num_filters = num_filters

    # check strides
    if(type(strides) != int):
      raise ValueError(" strides must be int ")
    self.strides = [1 , strides , strides , 1]

    # check the padding type
    if (padding in ["VALID" , "Valid" , "valid"]):
      self.padding = "VALID"
    elif (padding in ["same" , "SAME" , "Same"]) :
      self.padding = "SAME"
    else:
      raise ValueError("padding must be vaild or same")

    # Check activation function 
    if (activation == "relu"):
      self.activation = "relu"
    
    elif(activation == "sigmoid"):
      self.activation = "sigmoid"
    
    elif(activation == "tanh"):
      self.activation = "tanh"
    
    else:
      raise ValueError("{act} not define yet".format(act = activation))

  def apply(self , input):

    num_input_channels = input.get_shape()[-1]
    
    filters_shape = [self.kernel_size , self.kernel_size , num_input_channels , self.num_filters]
    # random weights
    weights = new_weights(shape = filters_shape)

    # Create new biases, one for each filter.
    biases = new_biases(length = self.num_filters)

    layer = tf.compat.v1.nn.conv2d(input=input , filter=weights , strides = self.strides , padding = self.padding)
    # add biases
    layer += biases

    if self.activation == "relu":
        layer = tf.nn.relu(layer)

    return layer

class Flatten:
  """
  Flatten layer 
  """
  def __init__(self ):
    pass
    
  def apply(self , layer):

    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()
    
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat

class Dense:
  """
  Function to add Fully connected layer
  """
  def __init__(self ,  neurouns , activation = "relu"):
    """
    neurons: number of neurons in the FC layer
    activation : activation funcion
    """
    if(type(neurouns) != int):
      raise ValueError(" numbers of neurouns must be int ")
    self.neurouns = neurouns
    # check for activation function 
    if (activation == "relu"):
      self.activation = "relu"
    
    elif(activation == "sigmoid"):
      self.activation = "sigmoid"
    
    elif(activation == "tanh"):
      self.activation = "tanh"
    
    else:
      raise ValueError("{act} not define yet".format(act = activation))

  
  def apply(self , input ): 
    num_inputs = input.shape[1]
    weights = new_weights(shape=[num_inputs , self.neurouns])
    biases = new_biases(length=self.neurouns)

    layer = tf.matmul(input, weights) + biases

    if self.activation == "relu":
        layer = tf.nn.relu(layer)

    return layer

class MaxPooling2D:
  """
  Maxpooling layer class 
  """

  def __init__(self , kernel_size = 2 , strides = 2 , padding = "valid"):
    """
    kernel_size : tuple or int of numbers
    strides     : strides wich want to apply
    padding     : padding of the input

    """    
    if ( type(kernel_size ) == int ):
      self.kernel_size = kernel_size
    elif (type(kernel_size) in [tuple , list] ):
      self.kernel_size = kernel_size[0] 
    else:
      raise ValueError("type of the kernel_size must be int ,  tuple or list")
    
    self.kernel_size = [1 , self.kernel_size , self.kernel_size , 1]

    # check if the strides is valid or not
    if ( type(strides ) == int ):
      self.strides = strides
    elif (type(strides) in [tuple , list] ):
      self.strides = strides[0] 
    else:
      raise ValueError("type of the kernel_size must be int ,  tuple or list")
    self.strides = [1 , self.strides , self.strides , 1]


    if (padding in ["VALID" , "Valid" , "valid"]):
      self.padding = "VALID"
    elif (padding in ["same" , "SAME" , "Same"]):
      self.padding = "SAME"
    else:
      raise ValueError("padding must be vaild or same")

  def apply(self , layer):
    return tf.compat.v1.nn.max_pool(value=layer ,
                                    ksize = self.kernel_size , strides  = self.strides ,
                                    padding = self.padding)
