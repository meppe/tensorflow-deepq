
# coding: utf-8

# In[16]:

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# In[17]:

import math
import random
import time

from collections import defaultdict



# In[19]:

from ..tf_rl.simulation import DoublePendulum
from tf_rl import simulate


# In[9]:

DOUBLE_PENDULUM_PARAMS = {
    'g_ms2': 9.8, # acceleration due to gravity, in m/s^2
    'l1_m': 1.0, # length of pendulum 1 in m
    'l2_m': 2.0, # length of pendulum 2 in m
    'm1_kg': 1.0, # mass of pendulum 1 in kg
    'm2_kg': 1.0, # mass of pendulum 2 in kg
    'damping': 0.4,
    'max_control_input': 20.0
}


# In[10]:

d = DoublePendulum(DOUBLE_PENDULUM_PARAMS)


# In[11]:

d.perform_action(0.2)


# In[13]:

try:
    simulate(d, fps=30, actions_per_simulation_second=1, speed=1.0, simulation_resultion=0.01)
except KeyboardInterrupt:
    print("Interrupted")


# In[ ]:



