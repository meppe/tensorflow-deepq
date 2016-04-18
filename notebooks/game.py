
# coding: utf-8

# In[4]:

# get_ipython().magic('load_ext autoreload')
# get_ipython().magic('autoreload 2')


# In[5]:

import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')


# In[6]:

import numpy as np
import tempfile
import tensorflow as tf

from tf_rl.controller import DiscreteDeepQ, HumanController
from tf_rl.simulation import KarpathyGame
from tf_rl import simulate
from tf_rl.models import MLP


# In[7]:

LOG_DIR = tempfile.mkdtemp()
print(LOG_DIR)


# In[8]:

from tf_rl.simulation import DiscreteHill


# In[9]:

# Tensorflow business - it is always good to reset a graph before creating a new controller.
tf.reset_default_graph()
session = tf.InteractiveSession()

# This little guy will let us run tensorboard
#      tensorboard --logdir [LOG_DIR]
journalist = tf.train.SummaryWriter(LOG_DIR)

# Brain maps from observation to Q values for different actions.
# Here it is a done using a multi layer perceptron with 2 hidden
# layers
brain = MLP([4,], [10, 4], 
            [tf.tanh, tf.identity])

# The optimizer to use. Here we use RMSProp as recommended
# by the publication
optimizer = tf.train.RMSPropOptimizer(learning_rate= 0.001, decay=0.9)

# DiscreteDeepQ object
current_controller = DiscreteDeepQ(4, 4, brain, optimizer, session,
                                   discount_rate=0.9, exploration_period=100, max_experience=10000, 
                                   store_every_nth=1, train_every_nth=4, target_network_update_rate=0.1,
                                   summary_writer=journalist)

session.run(tf.initialize_all_variables())
session.run(current_controller.target_network_update)
# graph was not available when journalist was created  
journalist.add_graph(session.graph_def)


# In[10]:

performances = []

try:
    for game_idx in range(2000):
        game = DiscreteHill()
        game_iterations = 0

        observation = game.observe()

        while game_iterations < 50 and not game.is_over():
            action = current_controller.action(observation)
            reward = game.collect_reward(action)
            game.perform_action(action)
            new_observation = game.observe()
            current_controller.store(observation, action, reward, new_observation)
            current_controller.training_step()
            observation = new_observation
            game_iterations += 1
        performance = float(game_iterations - (game.shortest_path)) / game.shortest_path
        performances.append(performance)
        if game_idx % 100 == 0:
            print "\rGame %d: iterations before success %d." % (game_idx, game_iterations),
            print "Pos: %s, Target: %s" % (game.position, game.target),
except KeyboardInterrupt:
    print "Interrupted"


# In[11]:

N = 500
smooth_performances = [float(sum(performances[i:i+N])) / N for i in range(0, len(performances) - N)]

plt.plot(range(len(smooth_performances)), smooth_performances)


# In[19]:

np.average(performances[-1000:])


# In[231]:

x = brain.layers[0].Ws[0].eval()
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')
plt.matshow(x)
plt.colorbar()


# In[138]:

brain.input_layer.b.eval()


# In[88]:

game.collect_reward(0)


# In[7]:

x = tf.Variable(tf.zeros((5,5)))


# In[8]:

tf.clip_by_norm(x, 5)


# In[ ]:



