from dqn_maker import dqn_maker
import ray


@ray.remote
class NNParamServer:
    def __init__(self):
        self.model = dqn_maker()

    def get_weights(self):
        return self.model.get_weights()

    def update_weights(self, gradient):
        self.model.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

    def update_weights_list(self, gradient_list):
        for gradient in gradient_list:
            self.update_weights(gradient)
