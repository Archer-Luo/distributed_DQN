hyperparam = {
    # environment parameters
    'state_dim': 2,
    'n_actions': 2,
    'start_state': [50, 100],
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
    'epi_len': 20,
    'batch_size': 200,
    'update_freq': 20,

    'eps_initial': 0.1,
    'eps_final': 0.1,
    'eps_final_state': 0.1,
    'eps_evaluation': 0.0,
    'replay_buffer_start_size': 8000,
    'eps_annealing_states': 250000,

    'eval': True,
    'eval_min': 400,
    'eval_start': [0, 0],
    'eval_len': 10000000,
    'eval_num': 50,
    'checkpoint': 'final_weights_3',
    'eval_freq': 200,

    'use_per': False,
    'priority_scale': 0.7,
    'offset': 0.1,

    'soft': False,
    'tau': 0.01,

    'clip': True,

    'initial_weights': 'final_weights',
    'optimum': 'result095'
}
