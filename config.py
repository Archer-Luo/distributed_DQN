hyperparam = {
    # environment parameters
    'state_dim': 2,
    'n_actions': 2,
    'start_state': [150, 150],
    'h': [3, 1],
    'rho': 0.75,
    'gamma': 0.998,

    # Learning parameters
    'nn_dimension': [20],
    'nn_activation': 'relu',
    'lr': 0.0001,

    'num_bundle': 20,

    'max_update_steps': 60000,
    'buffer_size': 500000,
    'C': 20,
    'epi_len': 100,
    'batch_size': 128,
    'update_freq': 4,

    'eps_initial': 0.1,
    'eps_final': 0.1,
    'eps_final_state': 0.1,
    'eps_evaluation': 0.0,
    'replay_buffer_start_size': 25000,
    'eps_annealing_states': 30000,

    'use_per': False,
    'priority_scale': 0.7,
    'offset': 0.1,

    'soft': False,
    'tau': 0.02,

    'clip': False,

    'initial_weights': './final_weights'
}
