hyperparam = {
    # environment parameters
    'state_dim': 2,
    'n_actions': 2,
    'start_state': [5, 5],
    'h': [3, 1],
    'rho': 0.8,

    # Learning parameters
    'nn_dimension': [10, 10, 10],
    'nn_activation': 'tanh',

    # 'num_workers': 8,
    # 'num_learners': 1,
    'num_bundle': 2,

    'max_update_steps': 100000,
    'sync_freq': 1,
    'C': 50,

    'gamma': 0.995,

    'eps_initial': 1,
    'eps_final': 0.1,
    'eps_final_state': 0.1,
    'eps_evaluation': 0.0,
    'replay_buffer_start_size': 512,
    'eps_annealing_states': 2000,

    'priority_scale': 0.7,

    # Buffer parameters
    'buffer_size': 10000,
    'actor_buffer_size': 1000,
    'batch_size': 128,
    'use_per': False,

    'offset': 0.1
}
