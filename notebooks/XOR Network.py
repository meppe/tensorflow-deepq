
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf

from __future__ import print_function


# # XOR Network

# ### Data generation

# In[2]:

def create_examples(N, batch_size):
    A = np.random.binomial(n=1, p=0.5, size=(batch_size, N))
    B = np.random.binomial(n=1, p=0.5, size=(batch_size, N,))

    X = np.zeros((batch_size, 2 *N,), dtype=np.float32)
    X[:,:N], X[:,N:] = A, B

    Y = (A ^ B).astype(np.float32)
    return X,Y


# In[3]:

X, Y = create_examples(3, 2)
print(X[0,:3], "xor", X[0,3:],"equals", Y[0])
print(X[1,:3], "xor", X[1,3:],"equals", Y[1])


# ### Xor cannot be solved with single layer of neural network

# In[22]:

import math

class Layer(object):
    def __init__(self, input_size, output_size):
        tensor_b = tf.zeros((output_size,))
        self.b = tf.Variable(tensor_b)
        tensor_W = tf.random_uniform((input_size, output_size),
                                     -1.0 / math.sqrt(input_size),
                                     1.0 / math.sqrt(input_size))
        self.W = tf.Variable(tensor_W)

    def __call__(self, x):
        return tf.matmul(x, self.W) + self.b


# In[105]:

tf.ops.reset_default_graph()
sess = tf.InteractiveSession()


# In[106]:

N = 5
# x represents input data
x = tf.placeholder(tf.float32, (None, 2 * N), name="x")
# y_golden is a reference output data.
y_golden = tf.placeholder(tf.float32, (None, N), name="y")

layer1 = Layer(2 * N, N)
# y is a linear projection of x with nonlinearity applied to the result.
y = tf.nn.sigmoid(layer1(x))

# mean squared error over all examples and all N output dimensions.
cost = tf.reduce_mean(tf.square(y - y_golden))

# create a function that will optimize the neural network
optimizer = tf.train.AdagradOptimizer(learning_rate=0.3)
train_op = optimizer.minimize(cost)

# initialize the variables
sess.run(tf.initialize_all_variables())


# In[107]:

for t in range(5000):
    example_x, example_y = create_examples(N, 10)
    cost_t, _ = sess.run([cost, train_op], {x: example_x, y_golden: example_y})
    if t % 500 == 0: 
        print(cost_t.mean())


# ### Notice that the error is far from zero.
# 
# Actually network always predicts approximately $0.5$, regardless of input data. That yields error of about $0.25$, because we use mean squared error ($0.5^2 = 0.25$). 

# In[109]:

X, _ = create_examples(N, 3)
prediction = sess.run([y], {x: X})
print(X)
print(prediction)


# ### Accuracy is not that hard to predict...

# In[113]:

N_EXAMPLES = 1000
example_x, example_y = create_examples(N, N_EXAMPLES)
# one day I need to write a wrapper which will turn the expression
# below to:
#     tf.abs(y - y_golden) < 0.5
is_correct = tf.less_equal(tf.abs(y - y_golden), tf.constant(0.5))
accuracy = tf.reduce_mean(tf.cast(is_correct, "float"))

acc_result = sess.run(accuracy, {x: example_x, y_golden: example_y})
print("Accuracy over %d examples: %.0f %%" % (N_EXAMPLES, 100.0 * acc_result))


# ### Xor Network with 2 layers

# In[149]:

tf.ops.reset_default_graph()
sess = tf.InteractiveSession()


# In[150]:

N = 5
# we add a single hidden layer of size 12
# otherwise code is similar to above
HIDDEN_SIZE = 12

x = tf.placeholder(tf.float32, (None, 2 * N), name="x")
y_golden = tf.placeholder(tf.float32, (None, N), name="y")

layer1 = Layer(2 * N, HIDDEN_SIZE)
layer2 = Layer(HIDDEN_SIZE, N) # <------- HERE IT IS!

hidden_repr = tf.nn.tanh(layer1(x))
y = tf.nn.sigmoid(layer2(hidden_repr))

cost = tf.reduce_mean(tf.square(y - y_golden))

optimizer = tf.train.AdagradOptimizer(learning_rate=0.3)
train_op = optimizer.minimize(cost)
sess.run(tf.initialize_all_variables())


# In[151]:

for t in range(5000):
    example_x, example_y = create_examples(N, 10)
    cost_t, _ = sess.run([cost, train_op], {x: example_x, y_golden: example_y})
    if t % 500 == 0: 
        print(cost_t.mean())


# ### This time the network works a tad better

# In[156]:

X, Y = create_examples(N, 3)
prediction = sess.run([y], {x: X})
print(X)
print(Y)
print(prediction)


# In[152]:

N_EXAMPLES = 1000
example_x, example_y = create_examples(N, N_EXAMPLES)
is_correct = tf.less_equal(tf.abs(y - y_golden), tf.constant(0.5))
accuracy = tf.reduce_mean(tf.cast(is_correct, "float"))

acc_result = sess.run(accuracy, {x: example_x, y_golden: example_y})
print("Accuracy over %d examples: %.0f %%" % (N_EXAMPLES, 100.0 * acc_result))

