import math
from typing import List, Tuple
# combine with schastic module pytorch?


from examples.vae import VAE, main
import torch.nn as nn
import torch
import storch
from torch.distributions import OneHotCategorical, Distribution, Normal, Bernoulli
from storch.method import (
    ScoreFunctionWOR,
    UnorderedSetEstimator,
    RELAX,
    REBAR,
    ScoreFunction,
    GumbelSoftmax,
    Expect,
)

K = 10

class KSubsetVAE(VAE):
    def initialize_param_layers(
        self, latents: int, prev_layer: int
    ) -> (nn.Module, nn.Module):
        fc3 = nn.Linear(prev_layer, 20)
        fc4 = nn.Linear(20, prev_layer)
        return fc3, fc4

    def initialize_method(self, args) -> storch.method.Method:
        if args.method == "lax":
            return storch.method.LAX("z", n_samples=args.samples, in_dim=args.latents)
        # Reinforce play with sampls and baseline type?
        elif args.method == "score":
            return storch.method.ScoreFunction(
                "z", n_samples=args.samples, baseline_factory=args.baseline
            )
        else:
            raise NotImplementedError()

    def prior(self, posterior: Normal):
        return Normal(loc=torch.zeros_like(posterior.loc), scale=torch.ones_like(posterior.scale))

    def variational_posterior(self, logits: torch.Tensor):
        return Normal(loc=logits, scale=torch.ones_like(logits))

    def logits_to_params(self, logits: torch.Tensor, latents: int) -> torch.Tensor:
        return logits

    def shape_latent(self, z: torch.Tensor, latents: int) -> torch.Tensor:
        i = torch.topk(z, 10, dim=-1).indices
        z = torch.zeros_like(z)
        z.scatter_(-1, i, 1)
        return z

    def name(self) -> str:
        return "normal-to-k-subset_vae"


class KSubsetDistribution(torch.distributions.ExponentialFamily):
    def __init__(self, logits: torch.Tensor, K: int):
        # See https://arxiv.org/pdf/2210.01941.pdf Appendix B
        super().__init__()
        self._bernoulli = Bernoulli(logits=logits)
        self.logits = logits
        self.pm = None
        self.K = K
        self._cache = {}
        self.partition = self.pr_exactly_k(0, self.logits.shape[-1], self.K)
        self.pm = {}

    def pr_exactly_k(self, l: int, u: int, k: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Algorithm 5 from paper
        if l > u:
            return 0.0
        if l == u:
            return self.logits[:, l].exp()
        if (l, u) in self._cache:
            return self._cache[(l, u)]
        pm = []
        for m in range(K + 1):
            fl_u = math.floor(u / 2)
            self._cache[l, fl_u] = self.pr_exactly_k(l, fl_u, m)
            self._cache[fl_u + 1, u] = self.pr_exactly_k(fl_u + 1, u, k - m)
            pm.append(self._cache[l, fl_u] * self._cache[fl_u + 1, u])
        self.pm[(l, u)] = pm

        return sum(pm)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        unnormalized_log_prob = self._bernoulli.log_prob(value)
        return unnormalized_log_prob - self.partition.log()

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            samp_2D = self._sample(sample_shape, 0, self.logits.shape[-1], self.K)

    def _sample(self, sample_shape, l: int, u: int, k: int):
        probs = self.pm[(l, u)].reshape(-1, K)
        m_star = torch.multinomial(probs, sample_shape.numel(), True)
        z_lower = self._sample((1,), l, math.floor(u / 2), m_star)
        z_upper = self._sample((1,), math.floor(u / 2) + 1, u, k - m_star)
        return torch.cat([z_lower, z_upper], dim=-1)

if __name__ == "__main__":
    main(KSubsetVAE)
