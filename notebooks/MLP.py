
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf

from __future__ import print_function


# In[2]:

def create_examples(N, batch_size):
    A = np.random.binomial(n=1, p=0.5, size=(batch_size, N))
    B = np.random.binomial(n=1, p=0.5, size=(batch_size, N,))

    X = np.zeros((batch_size, 2 *N,), dtype=np.float32)
    X[:,:N], X[:,N:] = A, B

    Y = (A ^ B).astype(np.float32)
    return X,Y


# In[3]:

import math

class Layer(object):
    def __init__(self, input_sizes, output_size):
        """Cretes a neural network layer."""
        if type(input_sizes) != list:
            input_sizes = [input_sizes]
        
        self.input_sizes = input_sizes
        self.output_size = output_size
                
        self.Ws = []
        for input_size in input_sizes:
            tensor_W = tf.random_uniform((input_size, output_size),
                                         -1.0 / math.sqrt(input_size),
                                         1.0 / math.sqrt(input_size))
            self.Ws.append(tf.Variable(tensor_W))

        tensor_b = tf.zeros((output_size,))
        self.b = tf.Variable(tensor_b)
            
    def __call__(self, xs):
        if type(xs) != list:
            xs = [xs]
        assert len(xs) == len(self.Ws),                 "Expected %d input vectors, got %d" % (len(self.Ws), len(x))
        return sum([tf.matmul(x, W) for x, W in zip(xs, self.Ws)]) + self.b

        
class MLP(object):
    def __init__(self, input_sizes, hiddens, nonlinearities):
        self.input_sizes = input_sizes
        self.hiddens = hiddens
        self.input_nonlinearity, self.layer_nonlinearities = nonlinearities[0], nonlinearities[1:]

        assert len(hiddens) == len(nonlinearities),                 "Number of hiddens must be equal to number of nonlinearities"
        
        self.input_layer = Layer(input_sizes, hiddens[0])
        self.layers = [Layer(h_from, h_to) for h_from, h_to in zip(hiddens[:-1], hiddens[1:])]

    def __call__(self, xs):
        if type(xs) != list:
            xs = [xs]
        hidden = self.input_nonlinearity(self.input_layer(xs))
        for layer, nonlinearity in zip(self.layers, self.layer_nonlinearities):
            hidden = nonlinearity(layer(hidden))
        return hidden


# In[19]:

tf.ops.reset_default_graph()
sess = tf.InteractiveSession()


# In[20]:

N = 5
# we add a single hidden layer of size 12
# otherwise code is similar to above
HIDDEN_SIZE = 12

x = tf.placeholder(tf.float32, (None, 2 * N), name="x")
y_golden = tf.placeholder(tf.float32, (None, N), name="y")

mlp = MLP(2 * N, [HIDDEN_SIZE, N], [tf.tanh, tf.sigmoid])
y = mlp(x)

cost = tf.reduce_mean(tf.square(y - y_golden))

optimizer = tf.train.AdagradOptimizer(learning_rate=0.3)
train_op = optimizer.minimize(cost)
sess.run(tf.initialize_all_variables())


# In[21]:

for t in range(5000):
    example_x, example_y = create_examples(N, 10)
    cost_t, _ = sess.run([cost, train_op], {x: example_x, y_golden: example_y})
    if t % 500 == 0: 
        print(cost_t.mean())


# In[22]:

N_EXAMPLES = 1000
example_x, example_y = create_examples(N, N_EXAMPLES)
is_correct = tf.less_equal(tf.abs(y - y_golden), tf.constant(0.5))
accuracy = tf.reduce_mean(tf.cast(is_correct, "float"))

acc_result = sess.run(accuracy, {x: example_x, y_golden: example_y})
print("Accuracy over %d examples: %.0f %%" % (N_EXAMPLES, 100.0 * acc_result))


# In[ ]:



