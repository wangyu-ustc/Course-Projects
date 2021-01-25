import torch
import numpy as np


def Wolfe_Powell(f, g, x, d, step=0.001, rho=0.25, sigma=0.15):
    # alpha_bar = step
    # while(f(x + alpha_bar * d) < f(x)):
    #     alpha_bar += step

    alpha_bar = 100

    # step (0)
    phi_0 = f(x)
    phi_0_d = torch.dot(g(x), d)
    a_1 = 0
    a_2 = alpha_bar
    phi_1 = phi_0
    phi_1_d = phi_0_d
    alpha = alpha_bar / 2

    # count = 0
    while(1):
        # step (1)
        phi = f(x + alpha*d)
        while(phi > phi_0 + rho * alpha * phi_0_d):
            alpha_hat = a_1 + (a_1 - alpha)**2 * phi_1_d / ( (phi_1 - phi)
                                - (a_1 - alpha) * phi_1_d) / 2
            a_2, alpha = alpha, alpha_hat
            phi = f(x + alpha*d)
        # step (2)
        phi_d = np.dot(g(x + alpha*d), d)
        if phi_d > sigma * phi_0_d:
            return alpha
        if phi_1_d - phi_d == 0:
            return alpha
        alpha_hat = alpha - (a_1 - alpha) * phi_d / (phi_1_d - phi_d)
        a_1 = alpha
        alpha = alpha_hat
        phi_1 = phi
        phi_1_d = phi_d

def Goldstein(f, g, x, d, step=0.001, rho=0.25, t=1.75):
    # alpha_bar = step
    # while (f(x + alpha_bar * d) < f(x)):
    #     alpha_bar += step

    alpha_bar = 100

    k = 0
    a, b = 0, alpha_bar
    alpha = alpha_bar / 2
    phi_0 = f(x)
    phi_0_d = torch.dot(g(x), d)

    while(1):
        phi = f(x + alpha * d)
        if(phi <= phi_0 + rho * alpha * phi_0_d + 1e-8):
        # 检验准则, 报告中的式(1); 若满足该式, 进入步3
            if(phi >= phi_0 + (1-rho) * alpha * phi_0_d):
                return alpha
            else:
                a = alpha
                if b > alpha_bar:
                    alpha = alpha * t
                    k += 1
                    continue
        else:
            b = alpha

        # 进入步4
        alpha = (a + b) / 2




