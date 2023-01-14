import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import random
from tqdm import tqdm_notebook as tqdm
import time
'''from Sinemodel import SineModel'''


import warnings

warnings.filterwarnings('ignore')

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class SineWaveTask:
    def __init__(self):
        self.a = np.random.uniform(0.1, 7.0)
        self.b = np.random.uniform(0, 2 * np.pi)
        self.train_x = None

    def f(self, x):
        return self.a * np.sin(x + self.b)

    def training_set(self, lower_bound=-7.0, upper_bound=7.0, size=14,
                     force_new=False):

        sample_step = (upper_bound - lower_bound) / size
        layer_lower_bound = np.arange(lower_bound, upper_bound, sample_step)
        layer_upper_bound = np.arange(lower_bound + sample_step, upper_bound + sample_step, sample_step)

        if self.train_x is None and not force_new:
            self.train_x = np.random.uniform(layer_lower_bound, layer_upper_bound,
                                             size)
            x = self.train_x
        elif not force_new:
            x = self.train_x
        else:
            x = np.random.uniform(layer_lower_bound, layer_upper_bound, size)
        y = self.f(x)
        return torch.Tensor(x), torch.Tensor(y)

    def test_set(self, size=1400):
        x = np.linspace(1, 3, size)
        y = self.f(x)
        return torch.Tensor(x), torch.Tensor(y)

    def plot(self, *args, **kwargs):
        x, y = self.test_set(size=100)
        return plt.plot(x.numpy(), y.numpy(), *args, **kwargs)


class ModifiableModule(nn.Module):
    def params(self):
        return [p for _, p in self.named_params()]

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self):
        subparams = []
        for name, mod in self.named_submodules():
            for subname, param in mod.named_params():
                subparams.append((name + '.' + subname, param))
        return self.named_leaves() + subparams

    def set_param(self, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in self.named_submodules():
                if module_name == name:
                    mod.set_param(rest, param)
                    break
        else:
            setattr(self, name, param)

    def icopy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = V(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class GradLinear(ModifiableModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)
        self.weights = V(ignore.weight.data, requires_grad=True)
        self.bias = V(ignore.bias.data, requires_grad=True)

    def forward(self, x):
        return F.linear(x, self.weights, self.bias)

    def named_leaves(self):
        return [('weights', self.weights), ('bias', self.bias)]


class SineModel(ModifiableModule):
    def __init__(self):
        super().__init__()
        self.hidden1 = GradLinear(1, 40)
        self.hidden2 = GradLinear(40, 40)
        self.out = GradLinear(40, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.out(x)

    def named_submodules(self):
        return [('hidden1', self.hidden1), ('hidden2', self.hidden2), ('out', self.out)]


TRAIN_SIZE = 160000
TEST_SIZE = 1000


SINE_TRAIN = [SineWaveTask() for _ in range(TRAIN_SIZE)]
SINE_TEST = [SineWaveTask() for _ in range(TEST_SIZE)]


def sine_fit1(net, wave, create_graph=False, force_new=False):
    net.train()

    x, y = wave.training_set(force_new=force_new)
    loss = F.mse_loss(net(V(x[:, None])), V(y).unsqueeze(1))
    loss.backward(create_graph=create_graph, retain_graph=True)

    return loss.data.cpu().numpy()


def meta_sgd(model, first_order=False, lr_inner=0.01):
    for name, param in model.named_params():
                grad = param.grad
                if first_order:
                    grad = V(grad.detach().data)
                model.set_param(name, param - lr_inner * grad)


def maml_sine(model, epochs, lr_inner=0.01, batch_size=1, first_order=False):
    optimizer = torch.optim.Adam(model.params())

    for _ in tqdm(range(epochs)):
        print(_+1)
        print(time.strftime('%Y-%m-%d %X ', time.localtime(time.time())))
        for i, t in enumerate(random.sample(SINE_TRAIN, len(SINE_TRAIN))):
            new_model = SineModel()
            new_model.icopy(model, same_var=True)

            loss = sine_fit1(new_model, t, create_graph=not first_order)

            meta_sgd(new_model, lr_inner=lr_inner)

            optimizer.zero_grad()

            sine_fit1(new_model, t, force_new=True)

            if (i + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()


SINE_MAML = SineModel()
maml_sine(SINE_MAML, 4)

torch.save(SINE_MAML, 'sine_learner_pycharm_try.pkl')
