from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tqdm import tqdm

class Model:

  def __init__(self ):
    """
    constractor of model
    inputs:
    img_size -> the input shape
    num_classes -> number of classes
    """
    # layers list
    self.Layers = []
    # number of classes    
    tf.compat.v1.disable_eager_execution()
    
    # placeholder to class while runung

  def add(self , layer ):
    #layer added to model
    self.Layers.append(layer)

  def Connect(self):
    last = self.X =self.Layers[0].apply()
    for i in range(1 , len(self.Layers)):
      last = self.Layers[i].apply(last)
    self.last = last

  
  def Compile(self , num_classes , loss = "softmax_cross_entropy" , optimizer = "adam" , learningRate = 0.001 ):
    """
    Compile function to set loss function and optimizer with init variables
    loss -> the loss function
    optimizer -> learning algorithm
    """
    self.Connect()
    
    self.learningRate = learningRate 
    
    # Check for optimizer type
    self.y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    
    #argmax for get class 
    self.y_true_cls = tf.argmax(self.y_true, axis=1) 
    self.y_pred = tf.nn.softmax(self.last)
    
    # Check loss function
    if(loss == "softmax_cross_entropy"):
      self.loss = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels = self.y_true , logits = self.y_pred ))
    else:
      raise ValueError("{opt} is not define yet".format(opt = loss))

    # Check your optimizer
    if(optimizer == 'GradientDescent' or optimizer == 'GDC'):
      self.optimizer = tf.train.GradientDescentOptimizer(self.learningRate).minimize(self.loss)
    elif(optimizer == 'Adam'):
      self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learningRate).minimize(self.loss)
    elif(optimizer == 'RMSProp'):
      self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)
    elif(optimizer == 'AdaGrad'):
      self.optimizer = tf.compat.v1.train.AdagradOptimizer(self.learningRate).minimize(self.loss)
    elif(optimizer == 'AdaGrad'):
      self.optimizer = tf.compat.v1.train.MomentumOptimizer(self.learningRate).minimize(self.loss)
    else:
      raise ValueError("{opt} is not define yet".format(opt = optimizer))

    self.y_pred_cls = tf.argmax(self.y_pred, axis=1)

    # check if pred_cls == true_cls
    self.correct_prediction = tf.equal(self.y_pred_cls , self.y_true_cls)
    # get the accuracy of model
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction , tf.float32))
    # init all vaiable in model
    self.init_op = tf.compat.v1.global_variables_initializer()
    # start the session
    self.sess = tf.compat.v1.Session() 
    # run the init
    self.sess.run(self.init_op)

  def fit(self , X , y , epochs , batch_size , validation_rate ):
    """
    fit the model 
    X: X_train data
    y: y_train data
    """
    #split data into train and validation 
    X_train ,  X_test , y_train  , y_test = train_test_split(X , y , test_size = validation_rate )
    # calculate the num of batches 
    total_batch = int(len(X_train) / batch_size)
    # loop through nums of epochs
    for epoch in range(epochs):
        avg_cost = 0
        avg_accuracy = 0
        # loop through the batches
        for i in range(total_batch):
            # calculate the batch size 
            batch_x  , batch_y = X_train[i * batch_size : (i + 1) * batch_size] , y_train[i * batch_size : (i + 1) * batch_size]
            
            _, c , acc = self.sess.run([self.optimizer , self.loss , self.accuracy ], feed_dict={self.X : batch_x ,self.y_true: batch_y})
            
            avg_cost += c / total_batch
            avg_accuracy += acc / total_batch
  
        print("Epoch:", (epoch + 1), "train_cost =", "{:.3f} ".format(avg_cost) , end = "")
        
        print("train_acc = {:.3f} ".format(avg_accuracy) , end = "")
        
        print("valid_acc = {:.3f}".format(self.sess.run(self.accuracy, feed_dict={self.X : X_test , self.y_true : y_test})))

  def predict(self , X ):
    """
    Function to return the predict (need to argmax)
    X : The input
    """
    return self.sess.run(self.y_pred , feed_dict = {self.X : X})
  
  def evaluate(self , X , y):
    """
    Function to return the accuracy of the model
    X : The input data
    y : the classes of the input
    """
    return self.sess.run(self.accuracy , feed_dict = {self.X : X , self.y_true : y})
  