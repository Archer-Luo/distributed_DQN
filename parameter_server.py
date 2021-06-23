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
        self.state_dim = hyperparam['state_dim']
        self.n_actions = hyperparam['n_actions']
        self.dimension = hyperparam['nn_dimension']
        weights = [xavier_numpy(self.state_dim, self.dimension[0])]
        for i in range(len(self.dimension) - 1):
            weights.append(xavier_numpy(self.dimension[i], self.dimension[i + 1]))
        weights.append(xavier_numpy(self.dimension[len(self.dimension) - 1], self.n_actions))

        list1 = weights
        list2 = []
        for i in self.dimension:
            list2.append(np.zeros(i))
        list2.append(np.zeros(self.n_actions))
        result = [None] * (len(list1) + len(list2))
        result[::2] = list1
        result[1::2] = list2

        self.weights = result
        self.update_step = 0

    def get_weights(self):
        return self.weights

    def update_weights(self, new_weights):
        [x + y for x, y in zip(self.weights, new_weights)]
        self.update_step += 1
        print(self.update_step, flush=True)

    def get_update_step(self):
        return self.update_step


# ray.init()
# parameter_server = ParamServer.remote()
# weights = parameter_server.get_weights.remote()
# print("done")
