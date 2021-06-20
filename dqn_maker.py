from tensorflow.python import keras
import tensorflow as tf
import config


def dqn_maker():
    state_dim = config['state_dim']
    n_actions = config['n_actions']
    activation = config['nn_activation']
    dimension = config['nn_dimension']

    model = keras.Sequential()
    model.add(tf.keras.Input(dimension=state_dim))
    for i in dimension:
        model.add(tf.keras.layers.Dense(i, activation=activation))
    model.add(tf.keras.layers.Dense(n_actions))

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam')

    return model
