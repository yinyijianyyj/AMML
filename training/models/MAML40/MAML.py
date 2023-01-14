import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
'''from Sinemodel import SineModel'''

import warnings

warnings.filterwarnings('ignore')


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


def sine_fit_points(net, train_x, train_y, optim=None, create_graph=False):
    net.train()
    if optim is not None:
        optim.zero_grad()
    x, y = train_x, train_y
    loss = F.mse_loss(net(V(x[:, None])), V(y).unsqueeze(1))
    loss.backward(create_graph=create_graph, retain_graph=True)
    if optim is not None:
        optim.step()


def copy_sine_model(model):
    m = SineModel()
    m.icopy(model)
    return m


def eval_sine_points(model, train_x, train_y, prid_x, fits=(0, 1), lr=0.01):
    x_prid = prid_x
    xtrain, ytrain = train_x, train_y

    points_model = copy_sine_model(model)
    optim = torch.optim.SGD(points_model.params(), lr)

    fit_res = []

    if 0 in fits:
        results = points_model(x_prid[:, None])
        fit_res.append((0, results))

    for i in range(np.max(fits)):
        sine_fit_points(points_model, xtrain, ytrain, optim)
        if i + 1 in fits:

            results = points_model(V(x_prid[:, None]))
            fit_res.append((i + 1, results))

            if i + 1 == np.max(fits):
                torch.save(points_model, 'meta_learner.pkl')

    return fit_res


def plot_sine_points(model, train_x, train_y, fits=(0, 1), lr=0.01):
    x_prid = torch.Tensor(np.linspace(0, 1, 1000))
    xtrain, ytrain = train_x, train_y

    fit_res = eval_sine_points(model, train_x, train_y, x_prid, fits, lr)

    train, = plt.plot(xtrain.numpy(), ytrain.numpy(), 'o', markeredgewidth=1.5, color='m',
                      markerfacecolor='white')

    plots = [train]
    legend = ['Training Points']
    for n, rst in fit_res:
        cur, = plt.plot(x_prid.numpy(), rst.cpu().data.numpy()[:, 0], '--')
        plots.append(cur)
        legend.append(f'After {n} Steps')
    plt.legend(plots, legend)
    plt.show()

    return rst.cpu().data.numpy()[:, 0]


x_train = torch.Tensor([0, 0.327, 0.5])
y_train = torch.Tensor([0, 0.47, 0.8])


sine_learner = torch.load('sine_learner.pkl')
a = plot_sine_points(sine_learner, x_train, y_train, fits=[0, 15, 2000], lr=0.005)
plt.show()
