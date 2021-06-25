from config import hyperparam
from replay_buffer import ReplayBuffer
from actor import Actor
from NN_parameter_server import NNParamServer
from learner import Learner
from dqn_maker import dqn_maker
import ray
import numpy as np


ray.init()
if __name__ == "__main__":
    parameter_server = NNParamServer.remote()
    final_weights = ray.get(parameter_server.get_weights.remote())
    np.save('final_weights.npy', final_weights)
