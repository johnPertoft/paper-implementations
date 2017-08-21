import tensorflow.contrib.distributions as ds


def input_tensor(batch_size, return_mixture=False):
    gaussians = [ds.MultivariateNormalDiag(loc=(5.0, 5.0), scale_diag=(0.5, 0.5)),
                 ds.MultivariateNormalDiag(loc=(-5.0, 5.0), scale_diag=(0.5, 0.5)),
                 ds.MultivariateNormalDiag(loc=(-5.0, -5.0), scale_diag=(0.5, 0.5)),
                 ds.MultivariateNormalDiag(loc=(5.0, -5.0), scale_diag=(0.5, 0.5))]
    uniform_mixture_probs = [1 / len(gaussians)] * len(gaussians)

    mixture = ds.Mixture(cat=ds.Categorical(uniform_mixture_probs),
                         components=gaussians)

    sampled = mixture.sample(batch_size)

    return (sampled, mixture) if return_mixture else sampled
