hyperparam = {
    # environment parameters
    'state_dim': 2,
    'n_actions': 2,
    'start_state': [5, 5],
    'h': [3, 1],
    'rho': 0.95,

    # Learning parameters
    'nn_dimension': [10, 10],
    'nn_activation': 'tanh',

    'num_bundle': 10,

    'max_update_steps': 50000,
    'C': 50,

    'gamma': 0.995,

    'eps_initial': 0.1,
    'eps_final': 0.1,
    'eps_final_state': 0.1,
    'eps_evaluation': 0.0,
    'replay_buffer_start_size': 20,
    'eps_annealing_states': 25000,

    'priority_scale': 0.7,

    # Buffer parameters
    'buffer_size': 10000,
    'batch_size': 20,
    'use_per': True,

    'offset': 0.1
}
