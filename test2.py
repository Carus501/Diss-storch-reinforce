import storch
import torch
from torch.distributions import Normal
from storch.method import Reparameterization, ScoreFunction

torch.manual_seed(0)

#
# def compute_f(method):
#     a = torch.tensor(5.0, requires_grad=True)
#     b = torch.tensor(-3.0, requires_grad=True)
#     c = torch.tensor(0.23, requires_grad=True)
#     d = a + b
#
#     # Sample e from a normal distribution using reparameterization
#     normal_distribution = Normal(b + c, 1)
#     e = method(normal_distribution)
#
#     f = d * e * e
#     return f, c
#
#
# repar = Reparameterization("e", n_samples=10)
# f, c = compute_f(repar)
#
# print(f.event_shape)
# f = f.view(1,10)
#
#
# storch.add_cost(f, "f")
# print(storch.backward())
# print("first derivative estimate", c.grad)
#
# f, c = compute_f(repar)
# storch.add_cost(f, "f")
# storch.backward()
# print(f)

x = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], [[6.0, 7.0, 8.0], [8.0, 9.0, 10.0]]])
y = torch.mean(x, dim=0)
print(x.shape)
print(y)
print(y.shape)
