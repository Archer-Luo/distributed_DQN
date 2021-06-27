hyperparam = {
    # environment parameters
    'state_dim': 2,
    'n_actions': 2,
    'start_state': [5, 5],
    'h': [3, 1],
    'rho': 0.85,

    # Learning parameters
    'nn_dimension': [100, 100],
    'nn_activation': 'relu',

    # 'num_workers': 8,
    # 'num_learners': 1,
    'num_bundle': 10,

    'max_update_steps': 50000,
    'C': 50,

    'gamma': 0.99,

    'eps_initial': 1,
    'eps_final': 0.1,
    'eps_final_state': 0,
    'eps_evaluation': 0.0,
    'replay_buffer_start_size': 1000,
    'eps_annealing_states': 25000,

    'priority_scale': 0.7,

    # Buffer parameters
    'buffer_size': 10000,
    'batch_size': 500,
    'use_per': True,

    'offset': 0.1
}
