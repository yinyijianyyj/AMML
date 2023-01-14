import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

import warnings

warnings.filterwarnings('ignore')

torch.manual_seed(0)


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


random_model = SineModel()


def copy_sine_model(model):
    m = SineModel()
    m.icopy(model)
    return m


def sine_fit_points(net, train_x, train_y, optim=None, create_graph=False):
    net.train()
    if optim is not None:
        optim.zero_grad()
    x, y = train_x, train_y
    loss = F.mse_loss(net(V(x[:, None])), V(y).unsqueeze(1))
    loss.backward(create_graph=create_graph, retain_graph=True)
    if optim is not None:
        optim.step()

    return loss.data.cpu().numpy()


def eval_sine_points(model, train_x, train_y, prid_x, fits=(0, 1), lr=0.01):
    x_prid = prid_x
    xtrain, ytrain = train_x, train_y

    points_model = copy_sine_model(model)
    optim = torch.optim.SGD(points_model.params(), lr)

    fit_res = []
    train_loss = []

    if 0 in fits:
        results = points_model(x_prid[:, None])
        fit_res.append((0, results))

    for i in range(np.max(fits)):
        loss = sine_fit_points(points_model, xtrain, ytrain, optim)
        if i + 1 in fits:

            results = points_model(V(x_prid[:, None]))
            fit_res.append((i + 1, results))
            train_loss.append(loss)

            if i + 1 == np.max(fits):
                '''torch.save(points_model, 'meta_learner.pkl')'''
                pass

    return fit_res, train_loss


def plot_sine_points(model, train_x, train_y, fits=(0, 1), lr=0.01):
    x_prid = torch.Tensor(np.linspace(0, 1, 1000))
    xtrain, ytrain = train_x, train_y

    fit_res, train_loss = eval_sine_points(model, train_x, train_y, x_prid, fits, lr)

    train, = plt.plot(xtrain.numpy(), ytrain.numpy(), 'o', markeredgewidth=1.5, color='m', markerfacecolor='white')

    plots = [train]
    legend = ['Training Points']
    fit_res_simple = [fit_res[0], fit_res[15], fit_res[np.max(fits)]]
    for n, rst in fit_res_simple:
        cur, = plt.plot(x_prid.numpy(), rst.cpu().data.numpy()[:, 0], '--')
        plots.append(cur)
        legend.append(f'After {n} Steps')
    plt.legend(plots, legend)
    plt.show()

    return rst.cpu().data.numpy()[:, 0], fit_res, train_loss


x_train = torch.Tensor([0, 0.327, 0.5])
y_train = torch.Tensor([0, 0.47, 0.8])

a, b, loss = plot_sine_points(random_model, x_train, y_train, fits=list(range(20002)), lr=0.005)
plt.show()
