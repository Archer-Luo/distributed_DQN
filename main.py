from config import hyperparam
from replay_buffer import ReplayBuffer
from actor import Actor
from parameter_server import ParamServer
from learner import Learner
import ray


ray.init()

if __name__ == "__main__":
    parameter_server = ParamServer.remote()
    replay_buffers = [ReplayBuffer.remote() for _ in range(hyperparam['num_bundle'])]
    actors = [Actor.remote(replay_buffer, parameter_server) for replay_buffer in replay_buffers]
    learners = [Learner.remote(replay_buffer, parameter_server) for replay_buffer in replay_buffers]
    for actor, learner in zip(actors, learners):
        actor.run.remote()
        learner.run.remote()
