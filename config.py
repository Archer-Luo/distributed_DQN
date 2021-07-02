hyperparam = {
    # environment parameters
    'state_dim': 2,
    'n_actions': 2,
    'start_state': [25, 25],
    'h': [3, 1],
    'rho': 0.95,

    # Learning parameters
    'nn_dimension': [100],
    'nn_activation': 'relu',

    'num_bundle': 10,

    'max_update_steps': 100000,
    'C': 50,
    'tau': 0.01,

    'gamma': 0.998,

    'eps_initial': 1,
    'eps_final': 0,
    'eps_final_state': 0,
    'eps_evaluation': 0.0,
    'replay_buffer_start_size': 200,
    'eps_annealing_states': 90000,

    'priority_scale': 0.7,

    # Buffer parameters
    'buffer_size': 50000,
    'batch_size': 200,
    'use_per': True,

    'offset': 0.1
}
