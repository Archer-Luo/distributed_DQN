config = {
    # environment parameters
    'rho': 0.8,

    # Learning parameters
    'num_workers': 8,
    'num_learners': 1,
    'num_step': 3,
    'batch_size': 512,
    'max_episode_steps': None,
    'param_update_interval': 50,
    'max_num_updates': 100000,
    'learning_rate': 0.0001,
    'gamma': 0.95,

    # Buffer parameters
    'buffer_max_size': 1000000,
    'use_per': True,
    'worker_buffer_size': 1000,
}
