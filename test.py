from config import hyperparam
from replay_buffer import ReplayBuffer
from actor import Actor
from parameter_server import ParamServer
from learner import Learner
import ray


ray.init()

if __name__ == "__main__":
    parameter_server = ParamServer.remote()
    replay_buffer = ReplayBuffer.remote()
    actor = Actor.remote(replay_buffer, parameter_server)
    actor.run.remote()
