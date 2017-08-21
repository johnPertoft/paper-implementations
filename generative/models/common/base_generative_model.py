

class BaseGenerativeModel:
    """Abstract Base class for generative models. Should not be instantiated."""

    def train_step(self, sess):
        raise NotImplementedError("No train_step(...) function defined.")

    def generate_results(self, sess, output_dir, param_settings):
        print("No generate_results(...) function defined.")
