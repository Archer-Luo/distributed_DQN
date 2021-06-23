from config import hyperparam
from replay_buffer import ReplayBuffer
from actor import Actor
from parameter_server import ParamServer
from learner import Learner
import ray


def main():
    parameter_server = ParamServer.remote()
    replay_buffers = []
    actors = []
    learners = []
    for _ in range(hyperparam['num_bundle']):
        replay_buffers.append(ReplayBuffer.remote())
    for replay_buffer in replay_buffers:
        actors.append(Actor.remote(replay_buffer, parameter_server))
        learners.append(Learner.remote(replay_buffer, parameter_server))
    for actor, learner in zip(actors, learners):
        actor.run.remote()
        learner.run.remote()


ray.init()

if __name__ == "__main__":
    main()
