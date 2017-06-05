import numpy as np
import tensorflow as tf

from generative.models import WGAN_GP


class TestWGAN(tf.test.TestCase):
    def test_fit(self):
        # TODO: Test to fit to a simple distribution. For now just building the model.
        X_sampled = tf.constant(np.random.randn(64, 50), dtype=tf.float32)
        wgan_gp = WGAN_GP(X_sampled, latent_dim=10, global_step=None)

        with self.test_session() as sess:
            pass
