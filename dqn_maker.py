from tensorflow.python import keras
import tensorflow as tf
from config import hyperparam


def dqn_maker():
    state_dim = hyperparam['state_dim']
    n_actions = hyperparam['n_actions']
    activation = hyperparam['nn_activation']
    dimension = hyperparam['nn_dimension']

    model = keras.Sequential()
    model.add(tf.keras.Input(dimension=state_dim))
    for i in dimension:
        model.add(tf.keras.layers.Dense(i, activation=activation))
    model.add(tf.keras.layers.Dense(n_actions))

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam')

    return model
