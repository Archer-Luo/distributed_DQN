from config import hyperparam
from replay_buffer import ReplayBuffer
import ray
from worker import Worker
import time
import numpy as np
from dqn_maker import dqn_maker
from NN_parameter_server import NNParamServer
import matplotlib.pyplot as plt
import tensorflow as tf


def main():
    parameter_server = NNParamServer.remote()
    replay_buffer = ReplayBuffer.remote()
    workers = [Worker.remote(i, replay_buffer, parameter_server) for i in range(hyperparam['num_bundle'])]
    ready_id, remaining_ids = ray.wait([worker.run.remote() for worker in workers], num_returns=hyperparam['num_bundle'])
    final_weights, record, outside, percentages = ray.get(ready_id[hyperparam['num_bundle'] - 1])

    plt.plot(percentages)
    plt.ylabel('percentages')
    plt.show()

    with open('record', 'w') as f:
        np.savetxt(f, record, fmt='%i', delimiter=",")

    print('outside: {0}'.format(outside), flush=True)

    evaluate_dqn = dqn_maker()
    evaluate_dqn.set_weights(final_weights)
    action_result = np.empty([151, 151])
    v_result = np.empty([151, 151])
    difference = np.empty([151, 151])

    evaluate_dqn.save_weights('final_weights_3')

    for a in range(151):
        for b in range(151):
            state = np.array([a, b])
            values = evaluate_dqn(np.expand_dims(state, axis=0), training=False).numpy().squeeze()
            action_result[a, b] = np.argmin(values) + 1
            difference[a, b] = values[0] - values[1]
            v_result[a, b] = np.amin(values)

    with open('rho{0}_gamma{1}_action'.format(hyperparam['rho'], hyperparam['gamma']), 'w') as f:
        np.savetxt(f, action_result, fmt='%i', delimiter=",")

    with open('difference', 'w') as f:
        np.savetxt(f, difference, fmt='%10.5f', delimiter=",")

    with open('rho{0}_gamma{1}_value'.format(hyperparam['rho'], hyperparam['gamma']), 'w') as f:
        np.savetxt(f, v_result, fmt='%10.5f', delimiter=",")


start_time = time.time()

ray.init(num_cpus=30, num_gpus=1)

if __name__ == "__main__":
    main()

print("--- %s seconds ---" % (time.time() - start_time))
