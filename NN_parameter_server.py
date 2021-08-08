from NmodelDynamics import ProcessingNetwork
from dqn_maker import dqn_maker
import ray
from config import hyperparam
import numpy as np
import scipy.stats
import tensorflow as tf


@ray.remote
def simulation(actions):
    eval_start = np.asarray(hyperparam['eval_start'])
    eval_len = hyperparam['eval_len']
    env = ProcessingNetwork.Nmodel_from_load(hyperparam['rho'])
    h = np.asarray(hyperparam['h'])

    total = 0.0
    current = eval_start
    for j in range(eval_len):
        total += current @ h
        clipped = np.minimum(current, np.full(2, 499))
        action = actions[clipped[0], clipped[1]]
        current = env.next_state_N1(current, action)
    return total / eval_len
#
#
# @ray.remote
# class Evaluator:
#     def __init__(self):
#         self.eval_start = np.asarray(hyperparam['eval_start'])
#         self.eval_len = hyperparam['eval_len']
#
#         self.model = dqn_maker()
#
#         self.env = ProcessingNetwork.Nmodel_from_load(hyperparam['rho'])
#         self.h = np.asarray(hyperparam['h'])
#         self.memory = np.zeros((500, 500))
#
#     def reset(self, weights):
#         self.model.set_weights(weights)
#         self.memory = np.zeros((500, 500))
#
#     async def simulation(self):
#         total = 0.0
#         current = self.eval_start
#         for j in range(self.eval_len):
#             if j % 1000 == 0:
#                 print(j)
#             total += current @ self.h
#             if current[0] < 500 and current[1] < 500:
#                 if self.memory[current[0], current[1]] != 0:
#                     action = self.memory[current[0], current[1]] - 1
#                 else:
#                     values = self.model(np.expand_dims(current, axis=0), training=False).numpy().squeeze()
#                     self.memory[current[0], current[1]] = np.argmin(values) + 1
#                     action = np.argmin(values) + 1
#             else:
#                 values = self.model(np.expand_dims(current, axis=0), training=False).numpy().squeeze()
#                 action = np.argmin(values) + 1
#             current = self.env.next_state_N1(current, action)
#         return total / self.eval_len


@ray.remote
class NNParamServer:
    def __init__(self):
        self.model = dqn_maker()
        self.sync_request = np.zeros(hyperparam['num_bundle'])
        self.param_weights = None
        self.processing = False
        self.count = 0
        self.correct = 0
        self.percentages = []
        self.update_num = 0

        self.eval_min = hyperparam['eval_min']
        self.eval_freq = hyperparam['eval_freq']
        self.eval_num = hyperparam['eval_num']
        self.t_val = scipy.stats.t.ppf(q=1 - 0.2 / 2, df=self.eval_num - 1)
        self.best_ceil = 2147483647
        self.best_fl = 2147483647
        self.best_sample = np.full(self.eval_num, fill_value=2147483647)
        self.checkpoint = hyperparam['checkpoint']

    def evaluation(self):
        actions = np.empty((500, 500))
        for a in range(500):
            for b in range(500):
                state = np.array([a, b])
                values = self.model(np.expand_dims(state, axis=0), training=False).numpy().squeeze()
                actions[a, b] = np.argmin(values)
        means = ray.get([simulation.remote(actions) for _ in range(self.eval_num)])
        means = np.asarray(means)
        stat, p = scipy.stats.ttest_ind(means, self.best_sample, equal_var=False)
        print('stat: ' + str(stat))
        print('p: ' + str(p))
        if p < 0.2:
            if stat > 0:
                self.model.load_weights(self.checkpoint)
                print('worse performance')
            elif stat < 0:
                self.best_sample = means
                self.model.save_weights(self.checkpoint)
                print('better performance')
        else:
            print('same performance')


        # new_mean = np.mean(means)
        # new_std = np.std(means)
        # new_ceil = new_mean + self.t_val * new_std
        # new_fl = new_mean - self.t_val * new_std
        # print('[' + str(new_fl) + ', ' + str(new_ceil) + ']')
        # if new_ceil < self.best_fl:
        #     self.best_ceil = new_ceil
        #     self.best_fl = new_fl
        #     self.model.save_weights(self.checkpoint)
        # elif new_fl > self.best_ceil:
        #     self.model.load_weights(self.checkpoint)
        # else:
        #     pass

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
                self.update_num += 1
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
            if self.update_num >= self.eval_min and self.update_num % self.eval_freq == 0:
                self.evaluation()
            self.param_weights = None
            if self.count != 0:
                percentage = self.correct / self.count * 100
                self.percentages.append(percentage)
                print(str(self.update_num) + ': ' + str(percentage))
            self.count = 0
            self.correct = 0

    def add_sample(self, num_sample, num_correct):
        self.count += num_sample
        self.correct += num_correct

    def get_percentages(self):
        return self.percentages

    # def get_errors(self):
    #     return ray.get(self.evaluator.get_errors.remote())
