from tensorflow.python import keras
import tensorflow as tf


def dqn_maker(n_actions):
    dqn = keras.Sequential([
        keras.layers.Dense(10, input_dim=2, activation='tanh'),
        keras.layers.Dense(10, activation='tanh'),
        keras.layers.Dense(10, activation='tanh'),
        keras.layers.Dense(n_actions)
    ])
    dqn.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam')

    return dqn
