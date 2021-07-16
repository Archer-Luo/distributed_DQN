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

# @ray.remote
# def evaluate(dqn, a, b):
#     state = np.array([a, b])
#     values = dqn.predict(np.expand_dims(state, axis=0)).squeeze()
#     action = np.argmax(values) + 1
#     difference = values[0] - values[1]
#     value = np.amax(values)
#     return [action, difference, value]


def main():
    parameter_server = NNParamServer.remote()
    replay_buffer = ReplayBuffer.remote()
    workers = [Worker.remote(i, replay_buffer, parameter_server) for i in range(hyperparam['num_bundle'])]
    ready_id, remaining_ids = ray.wait([worker.run.remote() for worker in workers], num_returns=1)
    final_weights, record, outside, percentages = ray.get(ready_id[0])

    plt.plot(percentages)
    plt.ylabel('percentages')
    plt.show()

    # with open('final_weights.txt', 'w') as f:
    #     print(final_weights, file=f)

    np.savetxt('record', record, fmt='%i', delimiter=",")
    print('outside: {0}'.format(outside))

    evaluate_dqn = dqn_maker()
    evaluate_dqn.set_weights(final_weights)
    action_result = np.empty([151, 151])
    v_result = np.empty([151, 151])
    difference = np.empty([151, 151])

    evaluate_dqn.save_weights('./final_weights')

    for a in range(151):
        for b in range(151):
            state = np.array([a, b])
            values = evaluate_dqn(np.expand_dims(state, axis=0), training=False).numpy().squeeze()
            action_result[a][b] = np.argmin(values) + 1
            difference[a][b] = values[0] - values[1]
            v_result[a][b] = np.amin(values)

    np.savetxt('rho{0}_gamma{1}_action'.format(hyperparam['rho'], hyperparam['gamma']),
               action_result, fmt='%i', delimiter=",")
    np.savetxt('difference', difference, fmt='%10.5f', delimiter=",")
    np.savetxt('rho{0}_gamma{1}_value'.format(hyperparam['rho'], hyperparam['gamma']),
               v_result, fmt='%10.5f', delimiter=",")


start_time = time.time()

ray.init(num_cpus=30, num_gpus=1)

if __name__ == "__main__":
    main()

print("--- %s seconds ---" % (time.time() - start_time))
