import numpy as np
from sklearn.metrics.pairwise import *
import sys
from scipy.linalg import svd
from math import *
import os
import scipy.linalg as spl
import torch
from .Config import *


def DAC_vec(X, lambda_, sample_size, kernel_function, kernel_param):

    n = X.shape[0]

    ind = np.arange(n)
    np.random.shuffle(ind)
    approximated_ls = np.zeros((n))

    # print(n/sample_size)
    true_sample_sizes = [min(sample_size, n - l*sample_size)
                         for l in range(0, int(np.ceil(n/sample_size)))]

    temp_inds = [ind[l*sample_size: l*sample_size + true_sample_sizes[l]]
                 for l in range(0, int(np.ceil(n/sample_size)) - 1)]
    var_ind = int(np.ceil(n/sample_size)) - 1
    temp_l = ind[var_ind*sample_size: var_ind *
                 sample_size + true_sample_sizes[var_ind]]

    Xs = torch.stack([X[temp_ind] for temp_ind in temp_inds])
    X_l = X[temp_l].reshape((1, true_sample_sizes[var_ind], -1))

    var = kernel_function(Xs, Xs, *kernel_param)
    var_l = kernel_function(X_l, X_l, *kernel_param)[0]

    # compute the approximated leverage score by inverting the small matrix
    res = (var * torch.linalg.inv(var + lambda_*torch.eye(sample_size,
           device=device).repeat(int(np.ceil(n/sample_size))-1, 1, 1))).sum(dim=2)
    res_l = (var_l * torch.linalg.inv(var_l + lambda_ *
             torch.eye(true_sample_sizes[var_ind], device=device))).sum(dim=1)

    for i, temp in enumerate(temp_inds):
        approximated_ls[temp] = res[i].cpu().detach().numpy()

    approximated_ls[temp_l] = res_l.cpu().detach().numpy()

    return approximated_ls


def DAC(X, lambda_, sample_size, kernel_function, kernel_param):
    """
    This function computes an approximation of the ridge leverage score, using a divide and conquer strategy.

    X: numpy array of size (n, d) where n is the number of data and d number of features.
    lambda_: regularisation term.
    sample_size: size of sub-matrix.
    kernel_function: a function that compute the kernel matrix, in the same form as the functions of the sklearn.metrics.pairwise library.
    kernel_param: the parameter of the kernel function (the degree if polynomial kernel for example).

    """

    n = X.shape[0]
    ind = np.arange(n)
    np.random.shuffle(ind)
    approximated_ls = np.zeros((n))

    # print(n/sample_size)

    for l in range(0, ceil(n/sample_size)):
        # sample a subset of data
        true_sample_size = min(sample_size, n - l*sample_size)

        temp_ind = ind[l*sample_size: l*sample_size + true_sample_size]
        # print(temp_ind)
        # compute the kernel matrix using the subset of selected data
        K_S = kernel_function(X[temp_ind], X[temp_ind], *kernel_param)
        # rint(K_S)

        # compute the approximated leverage score by inverting the small matrix
        approximated_ls[temp_ind] = np.sum(
            K_S * np.linalg.inv(K_S + lambda_ * np.eye(true_sample_size)), axis=1)

    return approximated_ls
