import numpy as np
from network_dict import network_dictionary
import itertools
import copy


class ProcessingNetwork:
    # N-model network class
    def __init__(self, A, D, alpha, mu, holding_cost, name):
        self.alpha = np.asarray(alpha)  # arrival rates
        self.mu = np.asarray(mu)  # service rates
        self.uniform_rate = np.sum(alpha) + np.sum(mu)  # uniform rate for uniformization
        self.p_arriving = np.divide(self.alpha, self.uniform_rate)  # normalized arrival rates
        self.p_compl = np.divide(self.mu, self.uniform_rate)  # normalized service rates
        self.cumsum_rates = np.unique(np.cumsum(np.concatenate([self.p_arriving, self.p_compl])))

        self.A = np.asarray(A)  # each row represents activity: -1 means job is departing, +1 means job is arriving
        self.D = np.asarray(D)  # ith row represents buffers that associated to the ith stations

        self.holding_cost = holding_cost
        self.network_name = name

    @classmethod
    def Nmodel_from_load(cls, load: float):
        # another constructor for the standard queuing networks
        # based on a queuing network name, find the queuing network info in the 'network_dictionary.py'
        return cls(A=network_dictionary['Nmodel']['A'],
                   D=network_dictionary['Nmodel']['D'],
                   alpha=np.asarray([1.3 * load, 0.4 * load]),
                   mu=network_dictionary['Nmodel']['mu'],
                   holding_cost=network_dictionary['Nmodel']['holding_cost'],
                   name=network_dictionary['Nmodel']['name'])

    def next_state_N1(self, state, action):
        """
        :param state: current state
        :param action: action
        :return: next state
        """

        w = np.random.random()
        wi = 0
        while w > self.cumsum_rates[wi]:
            wi += 1
        if wi == 0:
            state_next = state + np.asarray([1, 0])
        elif wi == 1:
            state_next = state + np.asarray([0, 1])
        elif wi == 2 and (state[0] > 0):
            state_next = state - np.asarray([1, 0])
        elif wi == 3 and ((action == 1 or state[1] == 0) and state[0] > 1):
            state_next = state - np.asarray([1, 0])
        elif wi == 4 and ((action == 0 or state[0] < 2) and state[1] > 0):
            state_next = state - np.asarray([0, 1])
        else:
            state_next = state
        return state_next
