from config import hyperparam
from replay_buffer import ReplayBuffer
from actor import Actor
from parameter_server import ParamServer
from learner import Learner
import ray
from worker import Worker


def main():
    parameter_server = ParamServer.remote()
    replay_buffer = ReplayBuffer.remote()
    workers = [Worker.remote(replay_buffer, parameter_server) for _ in range(hyperparam['num_bundle'])]
    return ray.get([worker.run.remote() for worker in workers])


ray.init()

if __name__ == "__main__":
    print(main())
