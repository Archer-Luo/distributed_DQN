import tensorflow as tf
from config import hyperparam


def dqn_maker():
    state_dim = hyperparam['state_dim']
    n_actions = hyperparam['n_actions']
    activation = hyperparam['nn_activation']
    dimension = hyperparam['nn_dimension']

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(state_dim,)))
    for i in dimension:
        model.add(tf.keras.layers.Dense(i, activation=activation))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(n_actions))

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005))  # TODO

    return model


# dqn = dqn_maker()
# trainable_var = dqn.trainable_variables
# weights = dqn.get_weights()
# print('done')
