import torch
import numpy as np

def Steepest_descent(f, g, h, x0, line_search, epsilon=0.001):
    x = x0
    k = 0
    d = -g(x)
    while(torch.norm(d) > epsilon):
        alpha = line_search(f, g, x, d)
        x = x + alpha * d
        d = -g(x)
        k += 1
        # print("iter [%d], value = %.8f" % (k, f(x)))
    return x, k


def Newton_descent(f, g, h, x0, line_search, epsilon=0.001):
    x = x0
    k = 0
    grad = g(x)
    d = - torch.mm(torch.inverse(h(x)), grad.unsqueeze(1)).reshape(-1)
    # print("gradient of first update:", g(x+d))
    while(torch.norm(grad) > epsilon):
        alpha = line_search(f, g, x, d)
        x = x + alpha * d
        grad = g(x)
        d = - torch.mm(torch.inverse(h(x)), grad.unsqueeze(1)).reshape(-1)
        k += 1
        # print("iter [%d], value = %.4f" % (k, f(x)))
    return x, k







