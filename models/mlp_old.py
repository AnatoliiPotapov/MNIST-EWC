import numpy as np
import tensorflow as tf

class Model:

    def __init__(self, l1 = 128, l2 = 128, l1_loss = 0.0, l2_loss = 0.0, sparse = False, ewc = 1e15, optimizer = 'Adam'):
        
        # Network Parameters
        n_hidden_1 = l1 # 1st layer number of features
        n_hidden_2 = l2 # 2nd layer number of features
        n_input = 784 # MNIST data input (img shape: 28*28)
        n_classes = 10 # MNIST total classes (0-9 digits)
        
        # tf Graph input
        self.tf_x = tf.placeholder("float", [None, n_input])
        self.tf_y = tf.placeholder("float", [None, n_classes])
        self.tf_labels = tf.placeholder("float", [None, n_classes])
        
        self.ewc = tf.constant(ewc)
        self.l1_loss = tf.constant(l1_loss)
        self.l2_loss = tf.constant(l2_loss)
        
        # Create model
        def multilayer_perceptron(x, weights):
            # Hidden layer with RELU activation
            layer_1 = tf.matmul(x, weights['h1'])
            layer_1 = tf.nn.relu(layer_1)
            # Hidden layer with RELU activation
            layer_2 = tf.matmul(layer_1, weights['h2'])
            layer_2 = tf.nn.relu(layer_2)
            # Output layer with linear activation
            out_layer = tf.matmul(layer_2, weights['out'])
            return out_layer
        
        self.multilayer_perceptron = multilayer_perceptron
        
        # Store layers weight
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h1e': tf.Variable(tf.random_normal([n_input, n_hidden_1]), trainable=False),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'h2e': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), trainable=False),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes])),
            'oute': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), trainable=False)
        }

        self.fisher = {
            'h1': tf.Variable(tf.zeros([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.zeros([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.zeros([n_hidden_2, n_classes]))
        }

        # Construct model
        self.pred = self.multilayer_perceptron(self.tf_x, self.weights)

        # Define loss and optimizer
        if sparse == False:
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.tf_y)) 
        if sparse == True:
            cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred, labels=self.tf_y)) 
        
        cost = cost + self.l1_loss*tf.reduce_sum(tf.abs(self.weights['h1']))
        cost = cost + self.l1_loss*tf.reduce_sum(tf.abs(self.weights['h2']))
        cost = cost + self.l1_loss*tf.reduce_sum(tf.abs(self.weights['h1']))
        cost = cost + self.l2_loss*tf.nn.l2_loss(self.weights['h1'])
        cost = cost + self.l2_loss*tf.nn.l2_loss(self.weights['h2'])
        cost = cost + self.l2_loss*tf.nn.l2_loss(self.weights['out'])
        self.cost = cost
        
        if optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(0.001)
        if optimizer == 'SGD':
            self.optimizer = tf.train.GradientDescentOptimizer(0.1)
        
        self.minimize_cost = self.optimizer.minimize(self.cost) 
        
        self.grads = self.optimizer.compute_gradients(self.cost, [self.weights['h1'], self.weights['h2'], self.weights['out']]) 

        ewc_cost = self.ewc * tf.nn.l2_loss(tf.multiply(self.fisher['h1'], tf.subtract(self.weights['h1'], self.weights['h1e'])))
        ewc_cost = ewc_cost + self.ewc * tf.nn.l2_loss(tf.multiply(self.fisher['h2'], tf.subtract(self.weights['h2'], self.weights['h2e'])))
        ewc_cost = ewc_cost + self.ewc * tf.nn.l2_loss(tf.multiply(self.fisher['out'], tf.subtract(self.weights['out'], self.weights['oute'])))
        self.ewc_cost = ewc_cost
        
        self.objective = self.cost + self.ewc_cost
        self.minimize_cost_ewc = self.optimizer.minimize(self.objective)
        
        # Initializing the variables
        self.init = tf.global_variables_initializer()        
     