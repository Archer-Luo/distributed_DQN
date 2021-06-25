hyperparam = {
    # environment parameters
    'state_dim': 2,
    'n_actions': 2,
    'start_state': [25, 25],
    'h': [3, 1],
    'rho': 0.8,

    # Learning parameters
    'nn_dimension': [10],
    'nn_activation': 'relu',

    # 'num_workers': 8,
    # 'num_learners': 1,
    'num_bundle': 8,

    'max_update_steps': 2500000,
    'C': 500,

    'gamma': 0.995,

    'eps_initial': 1,
    'eps_final': 0.1,
    'eps_final_state': 0.01,
    'eps_evaluation': 0.0,
    'replay_buffer_start_size': 100,
    'eps_annealing_states': 15000,

    'priority_scale': 0.7,

    # Buffer parameters
    'buffer_size': 40000,
    'batch_size': 10,
    'use_per': True,

    'offset': 0.1
}
