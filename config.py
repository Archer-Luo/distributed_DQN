hyperparam = {
    # environment parameters
    'state_dim': 2,
    'n_actions': 2,
    'start_state': [150, 150],
    'h': [3, 1],
    'rho': 0.95,
    'gamma': 0.998,

    # Learning parameters
    'nn_dimension': [20],
    'nn_activation': 'relu',
    'lr': 0.000001,

    'num_bundle': 20,

    'max_update_steps': 400000,
    'buffer_size': 1000000,
    'C': 100,
    'epi_len': 4000,
    'batch_size': 200,
    'update_freq': 20,

    'eps_initial': 0.5,
    'eps_final': 0.5,
    'eps_final_state': 0.5,
    'eps_evaluation': 0.0,
    'replay_buffer_start_size': 50000,
    'eps_annealing_states': 100000,

    'use_per': False,
    'priority_scale': 0.7,
    'offset': 0.1,

    'soft': False,
    'tau': 0.4,

    'clip': False,

    # 'initial_model': 'rho0.75_gamma0.998_model'
    # 'initial_model': 'final_model'

    'initial_weights': 'final_weights'
}
