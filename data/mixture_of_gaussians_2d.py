import tensorflow.contrib.distributions as ds
import numpy as np


def input_tensor(batch_size, n_gaussians=6, mixture_radius=5, stddev=0.1, return_mixture=False):
    theta = np.linspace(0, 2 * np.pi, n_gaussians + 1)[:-1]  # Skipping last because they're the same angle as first.
    gaussians = [ds.MultivariateNormalDiag(loc=(np.cos(t) * mixture_radius, np.sin(t) * mixture_radius),
                                           scale_diag=(stddev, stddev)) for t in theta]
    uniform_mixture_probs = [1 / len(gaussians)] * len(gaussians)

    mixture = ds.Mixture(cat=ds.Categorical(uniform_mixture_probs),
                         components=gaussians)

    sampled = mixture.sample(batch_size)

    return (sampled, mixture) if return_mixture else sampled
