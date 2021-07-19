import tensorflow as tf
from config import hyperparam


def dqn_maker():
    initial_weights = hyperparam['initial_weights']
    state_dim = hyperparam['state_dim']
    n_actions = hyperparam['n_actions']
    activation = hyperparam['nn_activation']
    dimension = hyperparam['nn_dimension']
    lr = hyperparam['lr']
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(state_dim,)))
    for i in dimension:
        model.add(tf.keras.layers.Dense(i, activation=activation, kernel_initializer=tf.keras.initializers.HeUniform()))
    model.add(tf.keras.layers.Dense(n_actions, kernel_initializer=tf.keras.initializers.HeUniform(), use_bias=False))

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

    if initial_weights is not None:
        model.load_weights(initial_weights)

    return model

