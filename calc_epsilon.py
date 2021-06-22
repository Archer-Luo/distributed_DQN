from config import hyperparam

# Epsilon information
eps_initial = hyperparam['eps_initial']
eps_final = hyperparam['eps_final']
eps_final_state = hyperparam['eps_final_state']
eps_evaluation = hyperparam['eps_evaluation']
replay_buffer_start_size = hyperparam['replay_buffer_start_size']
eps_annealing_states = hyperparam['eps_annealing_states']
max_update_steps = hyperparam['max_update_steps']

# Slopes and intercepts for exploration decrease
# (Credit to Fabio M. Graetz for this and calculating epsilon based on state number)
slope = -(eps_initial - eps_final) / eps_annealing_states
intercept = eps_initial - slope * replay_buffer_start_size
slope_2 = -(eps_final - eps_final_state) / (max_update_steps - eps_annealing_states - replay_buffer_start_size)
intercept_2 = eps_final_state - slope_2 * max_update_steps


def calc_epsilon(state_number, evaluation):
    """Get the appropriate epsilon value from a given state number
    Arguments:
        state_number: Global state number (used for epsilon)
        evaluation: True if the model is evaluating, False otherwise (uses eps_evaluation instead of default epsilon value)
    Returns:
        The appropriate epsilon value
    """
    if evaluation:
        return eps_evaluation
    elif state_number < replay_buffer_start_size:
        return eps_initial
    elif replay_buffer_start_size <= state_number < replay_buffer_start_size + eps_annealing_states:
        return slope * state_number + intercept
    elif state_number >= replay_buffer_start_size + eps_annealing_states:
        return slope_2 * state_number + intercept_2
