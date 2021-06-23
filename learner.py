import ray
import tensorflow as tf
import numpy as np
from config import hyperparam
from dqn_maker import dqn_maker
from calc_epsilon import calc_epsilon


@ray.remote
class Learner:

    def __init__(self, replay_buffer, param_server):
        self.replay_buffer = replay_buffer
        self.param_server = param_server

        self.dqn = dqn_maker()
        self.target_dqn = dqn_maker()

        self.max_update_steps = hyperparam['max_update_steps']
        self.C = hyperparam['C']

        # environment_info
        self.state_dim = hyperparam['state_dim']
        self.n_actions = hyperparam['n_actions']

        # buffer info
        self.use_per = hyperparam['use_per']
        self.priority_scale = hyperparam['priority_scale']
        self.batch_size = hyperparam['batch_size']

        # hyper-parameters
        self.gamma = hyperparam['gamma']

    def sync_dqn(self):
        new_weights = ray.get(self.param_server.get_weights.remote())
        self.dqn.set_weights(new_weights)

    def sync_target_dqn(self):
        new_weights = ray.get(self.param_server.get_weights.remote())
        self.target_dqn.set_weights(new_weights)

    def run(self):
        self.dqn.set_weights(ray.get(self.param_server.get_weights.remote()))
        self.target_dqn.set_weights(self.dqn.get_weights())

        while ray.get(self.replay_buffer.get_count.remote()) - 1 < self.batch_size:
            continue

        t = ray.get(self.param_server.get_update_step.remote())

        update_done = 0

        while t < self.max_update_steps:
            if update_done % self.sync_frequency == 0:
                self.sync_dqn()

            if update_done % self.C == 0:
                self.sync_target_dqn()

            if self.use_per:
                (states, actions, rewards, new_states,
                 terminal_flags), importance, indices = ray.get(self.replay_buffer.get_minibatch.remote(
                    priority_scale=self.priority_scale))
                importance = importance ** (1 - calc_epsilon(t, False))
            else:
                states, actions, rewards, new_states, terminal_flags = ray.get(self.replay_buffer.get_minibatch.remote(
                    batch_size=self.batch_size, priority_scale=self.priority_scale))

            # Target DQN estimates q-vals for new states
            target_future_v = np.amax(self.target_dqn.predict(new_states).squeeze(), axis=1)

            # Calculate targets (bellman equation)
            target_q = rewards + (self.gamma * target_future_v * (1 - terminal_flags))

            # Use targets to calculate loss (and use loss to calculate gradients)
            with tf.GradientTape() as tape:

                q_values = tf.squeeze(self.dqn(states))

                one_hot_actions = tf.keras.utils.to_categorical(actions, self.n_actions,
                                                                dtype=np.float32)  # using tf.one_hot causes strange errors
                Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)

                error = Q - target_q
                loss = tf.keras.losses.MeanSquaredError(target_q, Q)

                if self.use_per:
                    # Multiply the loss by importance, so that the gradient is also scaled.
                    # The importance scale reduces bias against situataions that are sampled
                    # more frequently.
                    loss = tf.reduce_mean(loss * importance)

            model_gradients = tape.gradient(loss, self.dqn.trainable_variables)
            gradients_numpy = []
            for variable in model_gradients:
                gradients_numpy.append(variable.numpy())

            self.param_server.update_weights.remote(gradients_numpy)

            if self.use_per:
                self.replay_buffer.set_priorities.remote(indices, error)

            update_done += 1
            t = ray.get(self.param_server.get_update_step.remote())

            print(float(loss.numpy()), flush=True)

        print("learner exits", flush=True)
