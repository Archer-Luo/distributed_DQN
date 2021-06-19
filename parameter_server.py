import tensorflow as tf
import ray
from tensorflow.python import keras
from dqn_maker import dqn_maker


@ray.remote
class ParamServer(object):
    def __init__(self, n_actions):
        self.model = dqn_maker(2)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def trainable_variables(self):
        return self.model.trainable_variables
