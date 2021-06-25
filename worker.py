import ray
import tensorflow as tf
import numpy as np
from config import hyperparam
from dqn_maker import dqn_maker
from calc_epsilon import calc_epsilon
from NmodelDynamics import ProcessingNetwork
import time
import random


@ray.remote
class Worker:

    def __init__(self, replay_buffer, param_server):
        self.replay_buffer = replay_buffer
        self.param_server = param_server

        self.env = ProcessingNetwork.Nmodel_from_load(hyperparam['rho'])
        self.h = np.array(hyperparam['h'])

        self.dqn = dqn_maker()
        self.target_dqn = dqn_maker()

        self.max_update_steps = hyperparam['max_update_steps']
        self.C = hyperparam['C']

        # environment_info
        self.state_dim = hyperparam['state_dim']
        self.n_actions = hyperparam['n_actions']

        # buffer info
        self.batch_size = hyperparam['batch_size']
        self.use_per = hyperparam['use_per']
        self.priority_scale = hyperparam['priority_scale']

        # hyper-parameters
        self.gamma = hyperparam['gamma']

        self.current_state = np.array(hyperparam['start_state'])
        self.t = 0

    def store(self, action, state, reward, terminal):
        self.replay_buffer.add_experience.remote(action, state, reward, terminal)

    def get_action(self, state_number, state, evaluation):
        eps = calc_epsilon(state_number, evaluation)

        # With chance epsilon, take a random action
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)

        # Otherwise, query the DQN for an action
        q_vals = self.dqn.predict(np.expand_dims(state, axis=0)).squeeze()
        action = q_vals.argmax()
        return action

    def sync_dqn(self):
        new_weights = ray.get(self.param_server.get_weights.remote())
        self.dqn.set_weights(new_weights)

    def sync_target_dqn(self):
        new_weights = ray.get(self.param_server.get_weights.remote())
        self.target_dqn.set_weights(new_weights)

    def run(self):
        time.sleep(random.randint(0, 10))
        self.target_dqn.set_weights(self.dqn.get_weights())
        self.dqn.set_weights(ray.get(self.param_server.get_weights.remote()))

        while self.t < self.max_update_steps:
            action = self.get_action(self.t, self.current_state, False)
            next_state = self.env.next_state_N1(self.current_state, action)
            reward = -(self.current_state @ self.h)
            terminal = (self.t == self.max_update_steps - 1)
            self.replay_buffer.add_experience.remote(action, self.current_state, reward, terminal)  # TODO

            self.current_state = next_state

            if ray.get(self.replay_buffer.get_count.remote()) - 1 < self.batch_size:
                continue

            self.sync_dqn()

            if self.use_per:
                (states, actions, rewards, new_states,
                 terminal_flags), importance, indices = ray.get(self.replay_buffer.get_minibatch.remote(
                    batch_size=self.batch_size, priority_scale=self.priority_scale))
                importance = importance ** (1 - calc_epsilon(self.t, False))
            else:
                states, actions, rewards, new_states, terminal_flags = ray.get(self.replay_buffer.get_minibatch.remote(
                    batch_size=self.batch_size, priority_scale=self.priority_scale))

            # Target DQN estimates q-vals for new states
            target_future_v = np.amax(self.target_dqn.predict(new_states).squeeze(), axis=1)

            # Calculate targets (bellman equation)
            target_q = rewards + (self.gamma * target_future_v * (1 - terminal_flags))

            # Use targets to calculate loss (and use loss to calculate gradients)
            with tf.GradientTape() as tape:

                q_values = self.dqn(states)

                one_hot_actions = tf.keras.utils.to_categorical(actions, self.n_actions,
                                                                dtype=np.float32)
                Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)

                error = Q - target_q
                loss = tf.keras.losses.MeanSquaredError()(target_q, Q)

                if self.use_per:
                    # Multiply the loss by importance, so that the gradient is also scaled.
                    # The importance scale reduces bias against situataions that are sampled
                    # more frequently.
                    loss = tf.reduce_mean(loss * importance)

            model_gradients = tape.gradient(loss, self.dqn.trainable_variables)
            # gradients_numpy = []
            # for variable in model_gradients:
            #     gradients_numpy.append(variable.numpy())

            self.param_server.update_weights.remote(model_gradients)

            if self.use_per:
                self.replay_buffer.set_priorities.remote(indices, error.numpy())

            self.t += 1

            if self.t % self.C == 0:
                self.sync_target_dqn()
                print(('{}' + ': ' + '{:10.5f}').format(self.t, loss), flush=True)

        return "done"
