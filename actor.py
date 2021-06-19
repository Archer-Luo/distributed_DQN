import time
import random
import tensorflow as tf
import ray
import numpy as np
from NmodelDynamics import ProcessingNetwork
from config import config
from dqn_maker import dqn_maker

@ray.remote
class Actor(object):

    def __init__(self, learner, replay_buffer, param_server):
        self.leaner = learner
        self.replay_buffer = replay_buffer
        self.param_server = param_server
        self.env = ProcessingNetwork.Nmodel_from_load(config['rho'])
        self.q_network = dqn_maker(2)

    def sync_with_param_server(self):
        new_weights = ray.get(self.param_server.getweights.remote())
        self.q_network.set_weights(new_weights)
