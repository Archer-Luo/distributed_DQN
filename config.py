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

    'max_update_steps': 100000,
    'buffer_size': 500000,
    'C': 20,
    'epi_len': 4000,
    'batch_size': 128,
    'update_freq': 4,

    'eps_initial': 1.0,
    'eps_final': 0.3,
    'eps_final_state': 0.01,
    'eps_evaluation': 0.0,
    'replay_buffer_start_size': 10000,
    'eps_annealing_states': 40000,

    'use_per': False,
    'priority_scale': 0.7,
    'offset': 0.1,

    'soft': False,
    'tau': 0.2,

    'clip': True,

    # 'initial_model': 'rho0.75_gamma0.998_model'
    # 'initial_model': 'final_model'

    'initial_weights': 'final_weights'
}
