hyperparam = {
    # environment parameters
    'state_dim': 2,
    'n_actions': 2,
    'start_state': [50, 50],
    'h': [3, 1],
    'rho': 0.75,
    'gamma': 0.998,

    # Learning parameters
    'nn_dimension': [20],
    'nn_activation': 'relu',
    'lr': 0.00001,

    'num_bundle': 10,

    'max_update_steps': 200000,
    'buffer_size': 1000000,
    'C': 1000,
    'epi_len': 1000,
    'batch_size': 32,
    'update_freq': 4,

    'eps_initial': 1.0,
    'eps_final': 0.1,
    'eps_final_state': 0.1,
    'eps_evaluation': 0.0,
    'replay_buffer_start_size': 10000,
    'eps_annealing_states': 50000,

    'use_per': True,
    'priority_scale': 0.7,
    'offset': 0.1,

    'soft': False,
    'tau': 0.02,

    'clip': True
}
