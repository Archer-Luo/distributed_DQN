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

    # 'num_workers': 8,
    # 'num_learners': 1,
    'num_bundle': 14,

    'max_update_steps': 20000,
    'C': 200,

    'gamma': 0.995,

    'eps_initial': 1,
    'eps_final': 0.1,
    'eps_final_state': 0.01,
    'eps_evaluation': 0.0,
    'replay_buffer_start_size': 200,
    'eps_annealing_states': 15000,

    'priority_scale': 0.7,

    # Buffer parameters
    'buffer_size': 10000,
    'batch_size': 50,
    'use_per': True,

    'offset': 0.1
}
