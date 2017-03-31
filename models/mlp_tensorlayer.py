import numpy as np
import tensorflow as tf
import tensorlayer as tl

class Model:
    """ This is fully-connected net with several hidden layers. """
    
    def __init__(self, neurons, losses, activation = 'relu', optimizer = 'Adam', biases = False, mask = False):
        
        tl.layers.set_name_reuse(True)
        self.graph = tf.Graph()
        
        with self.graph.as_default():
        
            l = len(neurons)

            # define input placeholder and layer
            self.x = tf.placeholder(tf.float32, shape=[None, neurons[0]], name='x')
            self.y_ = tf.placeholder(tf.int64, name='y_') 
            self.s = tf.placeholder(tf.float32, name='s') 
            
            self.network = tl.layers.InputLayer(self.x, name='input_layer') 

            f = {}
            w = {}
            
            if biases == False:
                bias_init = None
            if biases == True:
                bias_init = tf.constant_initializer(value=0.0)
                
            if activation == 'relu':
                activation = tf.nn.relu
            if activation == 'sigmoid':
                activation = tf.nn.sigmoid
  
             
            # define hidden layers
            for index, h_units in enumerate(neurons[1:-1]):
                self.network = tl.layers.DenseLayer(self.network, n_units=h_units,
                                    act=activation, b_init=bias_init, name='relu{0}'.format(index+1))
                
                # define w* and fisher coefs
                w['relu{0}/W'.format(index+1)] = tf.Variable(tf.random_normal([neurons[index], h_units]))
                f['relu{0}/W'.format(index+1)] = tf.Variable(tf.zeros([neurons[index], h_units]))
                
                if biases == True:
                        w['relu{0}/b'.format(index+1)] = tf.Variable(tf.zeros([h_units]))
                        f['relu{0}/b'.format(index+1)] = tf.Variable(tf.zeros([h_units]))

                
            # define output layer
            self.network = tl.layers.DenseLayer(self.network, n_units=neurons[l-1],
                                    act = tf.identity,
                                    b_init=bias_init,
                                    name='output_layer')
            
            w['output_layer/W'] = tf.Variable(tf.random_normal([neurons[l-2], neurons[l-1]]))
            f['output_layer/W'] = tf.Variable(tf.zeros([neurons[l-2], neurons[l-1]]))
            
            if biases == True:
                w['output_layer/b'.format(index+1)] = tf.Variable(tf.zeros([neurons[l-1]]))
                f['output_layer/b'.format(index+1)] = tf.Variable(tf.zeros([neurons[l-1]]))

            
            self.l1 = tf.constant(losses['l1'])
            self.l2 = tf.constant(losses['l2'])
            self.ewc = tf.constant(losses['ewc'])
            self.f = f
            self.w = w
            
            y = self.network.outputs
            
            # define cost function and metric
            if mask == False:
                self.cost = tl.cost.cross_entropy(y, self.y_, 'cost')
            if mask == True:
                yn = tf.multiply(y,self.s)
                self.cost = tl.cost.cross_entropy(yn, self.y_, 'cost')
                
            correct_prediction = tf.equal(tf.argmax(tf.multiply(y, self.s), 1), self.y_)
            self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.y_op = tf.argmax(tf.nn.softmax(y), 1)
            
            ewc_cost = self.cost
            
            # define ewc cost
            for m in self.network.all_params:
                name = m.name.split(":")[0]
                print(name, m.get_shape().as_list())
                ewc_cost = ewc_cost +  self.ewc * tf.nn.l2_loss( 
                    tf.multiply(self.f[name], tf.subtract(m, self.w[name])))
                
            self.ewc_cost = ewc_cost

            # define the optimizer
            train_params = self.network.all_params
            
            if optimizer == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(0.001)
            if optimizer == 'SGD':
                self.optimizer = tf.train.GradientDescentOptimizer(losses['lr'])
            
            self.train_op = self.optimizer.minimize(self.cost, var_list=train_params)
            self.train_op_ewc = self.optimizer.minimize(self.ewc_cost, var_list=train_params)

            # Initializing the variables
            self.init = tf.global_variables_initializer()  
            
            # Gradient op
            self.grads = self.optimizer.compute_gradients(self.cost, self.network.all_params)
            
