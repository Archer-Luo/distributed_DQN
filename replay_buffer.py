import os
import ray
import numpy as np
from config import hyperparam


@ray.remote
class ReplayBuffer:
    """Replay Buffer to store transitions.
    This implementation was heavily inspired by Fabio M. Graetz's replay buffer
    here: https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb"""
    def __init__(self):
        self.state_dim = hyperparam['state_dim']
        self.use_per = hyperparam['use_per']
        self.offset = hyperparam['offset']
        self.size = hyperparam['buffer_size']
        self.input_shape = (self.state_dim,)
        self.count = 0  # total index of memory written to, always less than self.size
        self.current = 0  # index to write to

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.uint8)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        self.states = np.empty((self.size, self.state_dim), dtype=np.uint8)
        self.priorities = np.zeros(self.size, dtype=np.float32)

    def get_count(self):
        return self.count

    def add_experience(self, action, state, reward, terminal):
        """Saves a transition to the replay buffer
        Arguments:
            action: An integer between 0 and env.action_space.n - 1
                determining the action the agent performed
            state: A (1, 2) state
            reward: A float determining the reward the agent received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if state.shape != self.input_shape:
            raise ValueError('Dimension of the state is wrong! state shape is %s and input_shape is %s' % (state.shape, self.input_shape,))

        # Write memory
        self.actions[self.current] = action
        self.states[self.current, ...] = state
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.priorities[self.current] = max(np.amax(self.priorities), 1.0)  # make the most recent experience important
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size

    def get_minibatch(self, priority_scale):
        """
        Returns:
            A tuple of states, actions, rewards, new_states, and terminals
            If use_per is True:
                An array describing the importance of transition. Used for scaling gradient steps.
                An array of each index that was sampled
        """

        if self.count - 1 < self.batch_size:
            raise ValueError('Not enough memories to get a minibatch')

        # Get sampling probabilities from priority list and get a list of indices
        if self.use_per:
            scaled_priorities = self.priorities[0:self.count-1] ** priority_scale
            sample_probabilities = scaled_priorities / sum(scaled_priorities)
            indices = np.random.choice(self.count-1, size=self.batch_size, replace=False, p=sample_probabilities)
            importance = 1 / self.count * 1 / sample_probabilities[indices]
            importance = importance / importance.max()
            # Retrieve states from memory
            states = self.states[indices, ...]
            new_states = self.states[indices + 1, ...]

            return (states, self.actions[indices], self.rewards[indices], new_states,
                    self.terminal_flags[indices]), importance, indices
        else:
            indices = np.random.choice(self.count-1, size=self.batch_size, replace=False)
            # Retrieve states from memory
            states = self.states[indices, ...]
            new_states = self.states[indices + 1, ...]

            return states, self.actions[indices], self.rewards[indices], new_states, self.terminal_flags[indices]

    def set_priorities(self, indices, errors):
        """Update priorities for PER
        Arguments:
            indices: Indices to update
            errors: For each index, the error between the target Q-vals and the predicted Q-vals
        """
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + self.offset

    def save(self, folder_name):
        """Save the replay buffer to a folder"""

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        np.save(folder_name + '/actions.npy', self.actions)
        np.save(folder_name + '/states.npy', self.states)
        np.save(folder_name + '/rewards.npy', self.rewards)
        np.save(folder_name + '/terminal_flags.npy', self.terminal_flags)

    def load(self, folder_name):
        """Loads the replay buffer from a folder"""
        self.actions = np.load(folder_name + '/actions.npy')
        self.states = np.load(folder_name + '/states.npy')
        self.rewards = np.load(folder_name + '/rewards.npy')
        self.terminal_flags = np.load(folder_name + '/terminal_flags.npy')
