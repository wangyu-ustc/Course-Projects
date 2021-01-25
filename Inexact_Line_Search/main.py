import time
import torch
import numpy as np
import argparse
from torch.autograd import Variable
from line_search import *
from gradident_descent import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2000, help='random seed')
    parser.add_argument('--dim', type=int, default=2, help='dimension of input')
    return parser.parse_args()

class Exp_Func(torch.nn.Module):
    def __init__(self, positive_definite=True):
        super(Exp_Func, self).__init__()
        if positive_definite == True:
            G = np.random.random((args.dim, args.dim))
            G = np.matmul(G, np.transpose(G))
        else:
            G = np.random.random((args.dim, args.dim)) + np.eye(args.dim)

        c = np.random.random(args.dim)
        bias = np.random.random()

        # 测试函数
        # G = np.array([[1.6989, 1.2013],[1.2013, 0.9552]])
        # c = np.array([0.0089, 0.9480])
        # bias = 0

        G,c,bias = torch.tensor(G, dtype=torch.float64), \
                   torch.tensor(c, dtype=torch.float64), torch.tensor(bias, dtype=torch.float64)
        self.G = torch.nn.Parameter(G)
        self.c = torch.nn.Parameter(c)
        self.bias = torch.nn.Parameter(bias)

    def forward(self, x):
        # return torch.matmul(self.linear(x), x.reshape(-1,1)).squeeze()
        return torch.dot(x, torch.matmul(self.G, x)) + torch.dot(x, self.c) + self.bias
    def Hesson(self):
        return 2 * self.G.detach()

if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)

    # Torch 自动求导
    # f = Exp_Func(positive_definite=True)
    # x0 = torch.tensor(np.random.random(args.dim), dtype=torch.float64)
    # def g(x):
    #     X = Variable(x, requires_grad=True)
    #     out = f(X)
    #     out.backward()
    #     return X.grad
    # def h(x):
    #     return f.Hesson()

    # 和上面相同, 不同的是这里是手动求导
    # x0 = torch.tensor(np.random.random(args.dim), dtype=torch.float64)
    # G = np.random.random((args.dim, args.dim)) + np.eye(args.dim)
    # c = np.random.random(args.dim)
    # bias = np.random.random()
    # f = lambda x: np.dot(x, np.matmul(G, x)) + np.dot(x, c)
    # g = lambda x: 2 * np.matmul(G, x) + c
    # h = lambda x: G

    x0 = torch.tensor([2, 2], dtype=torch.float64)
    f = lambda x: 100 * (x[0] - x[1] ** 2) ** 2 + (1 - x[1]) ** 2
    g = lambda x: torch.tensor([100 * 2 * (x[0] - x[1] ** 2), 100 * 2 * (x[0] - x[1] ** 2) * (-2 * x[1]) + 2 * (x[1] - 1)], dtype=torch.float64)
    h = lambda x: x


    # Fastest descend with  Line search
    print("############# Fastest Descend with Wolfe Powell ################")
    start = time.time()
    x, iter = Steepest_descent(f = f, g = g, h = h, x0 = x0, line_search=Wolfe_Powell)
    end = time.time()
    print("%.8f -> %.8f" % (f(x0), f(x)))
    print("iteration: %d" % iter)
    print("time consuming: %.5f" % (end-start))

    # Fastest descend with Goldstein Line search
    print("############# Fastest Descend with Goldstein ################")
    start = time.time()
    x, iter = Steepest_descent(f=f, g=g, h=h, x0=x0, line_search=Goldstein)
    end = time.time()
    print("%.8f -> %.8f" % (f(x0), f(x)))
    print("iteration: %d" % iter)
    print("time consuming: %.5f" % (end-start))


    print("############# Newton Descend with Wolfe Powell ################")
    # Newton descend with Wolfe_Powell Line search
    start = time.time()
    x, iter = Newton_descent(f = f, g = g, h=h, x0 = x0, line_search=Wolfe_Powell)
    end = time.time()
    print("%.8f -> %.8f" % (f(x0), f(x)))
    print("iteration: %d" % iter)
    print("time consuming: %.5f" % (end-start))


    print("############# Newton Descend with Goldstein ################")
    # Newton descend with Goldstein Line search
    start = time.time()
    x, iter = Newton_descent(f=f, g=g, h=h, x0=x0, line_search=Goldstein)
    end = time.time()
    print("%.8f -> %.8f" % (f(x0), f(x)))
    print("iteration: %d" % iter)
    print("time consuming: %.5f" % (end-start))








