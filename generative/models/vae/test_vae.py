import numpy as np
import tensorflow as tf

from generative.models import VAE


class TestGAN(tf.test.TestCase):
    def test_fit(self):
        # TODO: Test to fit to a simple distribution. For now just building the model.
        X_sampled = tf.constant(np.random.randn(64, 50), dtype=tf.float32)
        vae = VAE(X_sampled, latent_dim=10, global_step=None)

        with self.test_session() as sess:
            pass
