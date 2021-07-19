from NmodelDynamics import ProcessingNetwork
from dqn_maker import dqn_maker
import ray
from config import hyperparam
import numpy as np
import tensorflow as tf

#
# @ray.remote
# class Evaluator:
#     def __init__(self):
#         self.env = ProcessingNetwork.Nmodel_from_load(hyperparam['rho'])
#         self.h = np.asarray(hyperparam['h'])
#         self.states = np.asarray([[0, 0], [0, 50], [0, 100], [0, 150], [50, 0], [100, 0], [150, 0], [50, 50], [50, 100],
#                                   [100, 50], [100, 100], [150, 150]])
#         self.q_vals = np.asarray(
#             [9.42666, 9.958133333333334, 13.265379999999999, 17.221433333333334, 10.520326666666666,
#              13.176906666666667, 20.90934, 11.709173333333332, 14.09445333333333, 17.554286666666666,
#              21.03224,
#              37.82182])
#         self.immediate_actions = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
#         self.errors = []
#
#     def evaluate(self, weights):
#         evaluate_dqn = dqn_maker()
#         evaluate_dqn.set_weights(weights)
#         predictions = tf.squeeze(evaluate_dqn(self.states)).numpy()[range(len(self.states)), self.immediate_actions]
#         error = np.sum(np.absolute(predictions - self.q_vals))
#         self.errors.append(error)
#         print(error)
#
#     def get_errors(self):
#         return self.errors
#

@ray.remote
class NNParamServer:
    def __init__(self):
        self.model = dqn_maker()
        self.sync_request = np.zeros(hyperparam['num_bundle'])
        self.param_weights = None
        self.count = 0
        self.correct = 0
        self.percentages = []
        # self.evaluator = Evaluator.remote()

    def get_weights(self):
        return self.model.get_weights()

    def update_weights(self, gradient):
        self.model.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

    def update_weights_list(self, gradient_list):
        for gradient in gradient_list:
            self.update_weights(gradient)

    def sync(self, i):
        self.sync_request[i] = 1
        if np.isin(self.sync_request, [1, 2]).all():
            if self.param_weights is None:
                self.param_weights = self.model.get_weights()
                return self.param_weights
            else:
                return self.param_weights
        else:
            return None

    def confirm(self, i):
        self.sync_request[i] = 2
        if np.isin(self.sync_request, [2]).all():
            self.sync_request = np.zeros(hyperparam['num_bundle'])
            self.param_weights = None
            if self.count != 0:
                percentage = self.correct / self.count * 100
                self.percentages.append(percentage)
                print(percentage)
            self.count = 0
            self.correct = 0

    def add_sample(self, correct):
        self.count += 1
        if correct:
            self.correct += 1

    def get_percentages(self):
        return self.percentages

    # def get_errors(self):
    #     return ray.get(self.evaluator.get_errors.remote())
