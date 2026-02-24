#%%
import torch
import jax
import numpy as np
import jax.numpy as jnp
import scipy.linalg as spl

#%%


k=3


def recursiveNystrom(X, n_components, kernel_func, sigma,accelerated_flag=False, random_state=None, lmbda_0=0, return_leverage_score=True, **kwargs):
    
    """Function to compute RLS sampling scores and indices given points

    Returns:
        _type_: _description_
    """
    
    rng = np.random.RandomState(random_state)

    n_oversample = np.log(n_components)
    k = np.ceil(n_components / (4 * n_oversample)).astype(int)
    n_levels = np.ceil(np.log(X.shape[0] / n_components) / np.log(2)).astype(int)
    perm = rng.permutation(X.shape[0])

    # set up sizes for recursive levels
    size_list = [X.shape[0]]
    for l in range(1, n_levels+1):
        size_list += [np.ceil(size_list[l - 1] / 2).astype(int)]

    # indices of poitns selected at previous level of recursion
    # at the base level it's just a uniform sample of ~ n_component points
    sample = np.arange(size_list[-1])
    indices = perm[sample]
    weights = np.ones((indices.shape[0],))

    # we need the diagonal of the whole kernel matrix, so compute upfront
    k_diag = np.ones((X.shape[0], 1)) #kernel_func(X,X,sigma)

    # Main recursion, unrolled for efficiency
    for l in reversed(range(n_levels)):
        # indices of current uniform sample
        current_indices = perm[:size_list[l]]
        # build sampled kernel

        # all rows and sampled columns
        KS = kernel_func(X[current_indices,:], X[indices,:],*sigma)
        #print(KS.shape)
        SKS = KS[sample, :] # sampled rows and sampled columns
        #print(SKS.shape)

        # optimal lambda for taking O(k log(k)) samples
        if k >= SKS.shape[0]:
            # for the rare chance we take less than k samples in a round
            lmbda = 10e-6
            # don't set to exactly 0 to avoid stability issues
        else:
            # eigenvalues equal roughly the number of points per cluster, maybe this should scale with n?
            # can be interpret as the zoom level
            lmbda = (np.sum(np.diag(SKS) * (weights ** 2))
                    - np.sum(spl.eigvalsh(SKS * weights[:,None] * weights[None,:], eigvals=(SKS.shape[0]-k, SKS.shape[0]-1))))/k
        lmbda = np.maximum(lmbda_0*SKS.shape[0], lmbda)
        if lmbda == lmbda_0*SKS.shape[0]:
            print("Set lambda to %d." % lmbda)
        #lmbda = np.minimum(lmbda, 5)
            # lmbda = spl.eigvalsh(SKS * weights * weights.T, eigvals=(0, SKS.shape[0]-k-1)).sum()/k
            # calculate the n-k smallest eigenvalues

        # compute and sample by lambda ridge leverage scores
        R = np.linalg.inv(SKS + np.diag(lmbda * weights ** (-2)))
        R = np.matmul(KS, R)
        #R = np.linalg.lstsq((SKS + np.diag(lmbda * weights ** (-2))).T,KS.T)[0].T
        if l != 0:
            # max(0, . ) helps avoid numerical issues, unnecessary in theory
            leverage_score = np.minimum(1.0, n_oversample * (1 / lmbda) * np.maximum(+0.0, (
                    k_diag[current_indices, 0] - np.sum(R * KS, axis=1))))
            # on intermediate levels, we independently sample each column
            # by its leverage score. the sample size is n_components in expectation
            sample = np.where(rng.uniform(size=size_list[l]) < leverage_score)[0]
            # with very low probability, we could accidentally sample no
            # columns. In this case, just take a fixed size uniform sample
            if sample.size == 0:
                leverage_score[:] = n_components / size_list[l]
                sample = rng.choice(size_list[l], size=n_components, replace=False)
            weights = np.sqrt(1. / leverage_score[sample])

        else:
            leverage_score = np.minimum(1.0, (1 / lmbda) * np.maximum(+0.0, (
                    k_diag[current_indices, 0] - np.sum(R * KS, axis=1))))
            p = leverage_score/leverage_score.sum()

            sample = rng.choice(X.shape[0], size=n_components, replace=False, p=p)
        indices = perm[sample]

    if return_leverage_score:
        return indices, leverage_score[np.argsort(perm)]
    else:
        return indices
# %%
