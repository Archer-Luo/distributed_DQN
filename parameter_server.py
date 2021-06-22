import ray
from config import hyperparam
import numpy as np
import math


def xavier_numpy(fan_in, fan_out):
    limit = math.sqrt(6 / (fan_in + fan_out))
    weights = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
    return weights


@ray.remote
class ParamServer:
    def __init__(self):
        n_actions = hyperparam['n_actions']
        dimension = hyperparam['nn_dimension']
        weights = [xavier_numpy(n_actions, dimension[0])]
        for i in range(len(dimension) - 1):
            weights.append(xavier_numpy(dimension[i], dimension[i + 1]))
        weights.append(xavier_numpy(dimension[len(dimension) - 1], n_actions))

        self.weights = weights
        self.update_step = 0

    def get_weights(self):
        return self.weights

    def update_weights(self, new_weights):
        [x + y for x, y in zip(self.weights, new_weights)]
        self.update_step += 1
        print(self.update_step)

    def get_update_step(self):
        return self.update_step
