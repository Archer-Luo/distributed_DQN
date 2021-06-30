from config import hyperparam
from replay_buffer import ReplayBuffer
import ray
from worker import Worker
import time
import numpy as np
from dqn_maker import dqn_maker
from NN_parameter_server import NNParamServer


def main():
    # parameter_server = ParamServer.remote()
    parameter_server = NNParamServer.remote()
    replay_buffer = ReplayBuffer.remote()
    workers = [Worker.remote(replay_buffer, parameter_server) for _ in range(hyperparam['num_bundle'])]
    ready_id, remaining_ids = ray.wait([worker.run.remote() for worker in workers], num_returns=1)
    final_weights = ray.get(ready_id[0])

    with open('final_weights.txt', 'w') as f:
        print(final_weights, file=f)

    evaluate_dqn = dqn_maker()
    evaluate_dqn.set_weights(final_weights)
    action_result = np.empty([50, 50])
    v_result = np.empty([50, 50])
    for a in range(50):
        for b in range(50):
            state = np.array([a, b])
            values = evaluate_dqn.predict(np.expand_dims(state, axis=0)).squeeze()
            action_result[a][b] = np.argmax(values) + 1
            v_result[a][b] = np.amax(values)
    np.savetxt('rho{0}_gamma{1}_action'.format(hyperparam['rho'], hyperparam['gamma']),
               action_result, fmt='%i', delimiter=",")
    np.savetxt('rho{0}_gamma{1}_value'.format(hyperparam['rho'], hyperparam['gamma']),
               v_result, fmt='%10.5f', delimiter=",")


start_time = time.time()

ray.init(num_cpus=5, num_gpus=1)

if __name__ == "__main__":
    main()

print("--- %s seconds ---" % (time.time() - start_time))
