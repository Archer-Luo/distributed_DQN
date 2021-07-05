hyperparam = {
    # environment parameters
    'state_dim': 2,
    'n_actions': 2,
    'start_state': [25, 25],
    'h': [3, 1],
    'rho': 0.95,

    # Learning parameters
    'nn_dimension': [10, 10, 10],
    'nn_activation': 'tanh',

    'num_bundle': 20,

    'max_update_steps': 80000,
    'C': 25,
    'epi_len': 200,

    'gamma': 0.998,

    'eps_initial': 1,
    'eps_final': 1,
    'eps_final_state': 1,
    'eps_evaluation': 0.0,
    'replay_buffer_start_size': 200,
    'eps_annealing_states': 19000,

    'priority_scale': 0.7,

    # Buffer parameters
    'buffer_size': 50000,
    'batch_size': 200,
    'use_per': True,

    'offset': 0.1
}
