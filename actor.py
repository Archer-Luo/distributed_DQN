import time
import ray
import numpy as np
from NmodelDynamics import ProcessingNetwork
from config import hyperparam
from dqn_maker import dqn_maker
from calc_epsilon import calc_epsilon


@ray.remote
class Actor:

    def __init__(self, replay_buffer, param_server):
        self.replay_buffer = replay_buffer
        self.param_server = param_server
        self.env = ProcessingNetwork.Nmodel_from_load(hyperparam['rho'])
        self.q_network = dqn_maker()

        self.actor_buffer_size = hyperparam['actor_buffer_size']  # TODO

        self.max_update_steps = hyperparam['max_update_steps']
        self.sync_freq = hyperparam['sync_freq']

    def sync_with_param_server(self):
        new_weights = self.param_server.get_weights.remote()
        self.q_network.set_weights(new_weights)

    def store(self, action, state, reward, terminal):
        self.replay_buffer.add_experience.remote(action, state, reward, terminal)

    def get_action(self, state_number, state, evaluation):
        """Query the DQN for an action given a state
        Arguments:
            state_number: Global state number (used for epsilon)
            state: State to give an action for
            evaluation: True if the model is evaluating, False otherwise (uses eps_evaluation instead of default epsilon value)
        Returns:
            An integer as the predicted move
        """

        # Calculate epsilon based on the state number
        eps = calc_epsilon(state_number, evaluation)

        # With chance epsilon, take a random action
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)

        # Otherwise, query the DQN for an action
        q_vals = self.q_network.predict(np.expand_dims(state, axis=0)).squeeze()
        action = q_vals.argmax()
        return action

    def run(self):
        start_state = np.array(hyperparam['start_state'])

        current_state = np.copy(start_state)
        h = np.array(hyperparam['h'])
        t = self.param_server.get_update_step.remote()

        experience_generated = 0

        while t < self.max_update_steps:
            if experience_generated % self.sync_freq == 0:
                self.sync_with_param_server()

            action = self.get_action(t, current_state, False)
            next_state = self.env.next_state_N1(current_state, action)
            reward = -(current_state @ h)
            terminal = (t == self.max_update_steps - 1)
            self.replay_buffer.add_experience.remote(action, current_state, reward, terminal)

            experience_generated += 1

            current_state = next_state
            t = self.param_server.get_update_step.remote()

        print('actor exits')




