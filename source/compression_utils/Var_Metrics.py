import jax
import jax.numpy as jnp
from pykeops.torch import LazyTensor
from .Config import *
from .preproc import var_extract, var_proc


if kernel_name == 'gaussian':

    def spherical(ss, gamma):

        return (-(2 - 2*ss)*gamma).exp()

    def spherical_1(ss, gamma):
        return jnp.exp(-(2 - 2*ss)*gamma)

if kernel_name == 'binet':
    def spherical(ss, gamma):
        return ss**2

    def spherical_1(ss, gamma):
        return ss**2


@jax.jit
def kern_m(XX, YY, sigma, sig):
    """Function to compue kernel matrix of Varifolds using JAX

    Args:
        XX (_type_): _description_
        YY (_type_): _description_
        sigma (_type_): _description_
        sig (_type_): _description_

    Returns:
        _type_: _description_
    """
    gamma = 1 / (2*(sigma * sigma))
    gamma_1 = 1/(2*(sig*sig))

    X = XX[:, :k].reshape((1, -1, k))
    Y = YY[:, :k].reshape((1, -1, k))
    u = XX[:, k:][:, None, :]
    v = YY[:, k:][None, :, :]
    # print(u.shape)
    ZZ = jnp.sum(X*X, axis=2, keepdims=True) + jnp.sum(Y*Y,
                                                       axis=2, keepdims=True).transpose([0, 2, 1])
    SS = -2*X@Y.transpose([0, 2, 1])

    ss = ((u*v)).sum(-1)

    res = spherical_1(ss, gamma_1)
    K = jnp.exp(- (jnp.abs(SS + ZZ))/(2*(sigma**2)))

    return (K*res)[0]
# REFACTOR


def Var_met(x, y, u, v, sigma, sig):
    """Function to compute kernel matrix of Varifolds metric
     using Keops

  Args:
      x (_type_): _description_
      y (_type_): _description_
      u (_type_): _description_
      v (_type_): _description_
      sigma (_type_): _description_
      sig (_type_): _description_

  Returns:
      _type_: _description_
  """
    # x, y, b = Vi(0, 3), Vj(1, 3), Vj(2, 3)
    x = LazyTensor(x[:, None, :])
    y = LazyTensor(y[None, :, :])

    u = LazyTensor(u[:, None, :])
    v = LazyTensor(v[None, :, :])
    gamma = 1 / (2*(sigma * sigma))
    gamma_1 = 1/(2*(sig*sig))

    D2 = x.sqdist(y)
    ss = ((u*v)).sum()
    res = spherical(ss, gamma_1)
    K = (-D2 * gamma).exp()
    return res*K


def Var_met_reduce(x, y, u, v, b, sigma, sig):
    """Function to compute vector field of Varifols

  Args:
      x (_type_): _description_
      y (_type_): _description_
      u (_type_): _description_
      v (_type_): _description_
      b (_type_): _description_
      sigma (_type_): _description_
      sig (_type_): _description_

  Returns:
      _type_: _description_
  """
    # x, y, b = Vi(0, 3), Vj(1, 3), Vj(2, 3)
    x = LazyTensor(x[:, None, :])
    y = LazyTensor(y[None, :, :])

    u = LazyTensor(u[:, None, :])
    v = LazyTensor(v[None, :, :])
    gamma = 1 / (2*(sigma * sigma))
    gamma_1 = 1/(2*(sig*sig))

    D2 = x.sqdist(y)
    ss = ((u*v)).sum()
    # print(ss.shape)
    res = spherical(ss, gamma_1)
    K = (-D2 * gamma).exp()

    return (res*K)@b  # res*K

# what is this used for


def kern_met_new(XX, YY, sigma, sig):
    """Function to compue kernel matrix of Varifolds using JAX

    Args:
        XX (_type_): _description_
        YY (_type_): _description_
        sigma (_type_): _description_
        sig (_type_): _description_

    Returns:
        _type_: _description_
    """
    gamma = 1 / (2*(sigma * sigma))
    gamma_1 = 1/(2*(sig*sig))

    X = XX[:, :, :k]  # .reshape((1, -1, k))
    Y = YY[:, :, :k]  # .reshape((1, -1, k))
    u = XX[:, :, k:]  # XX[:,:,None, k:]#[:, None, :]
    v = YY[:, :, k:]  # YY[:,None,:, k:]#[None, :, :]
    # print(u.shape)
    A = torch.sum(X*X, dim=2, keepdim=True)
    B = torch.sum(Y*Y, dim=2, keepdim=True).transpose(2, 1)
    C = - 2*X@Y.transpose(2, 1)

    ss = torch.einsum('abc,aec->abe', u, v)

    # del ss
    # print(SS.shape,ZZ.shape)

    res = spherical(ss, gamma_1)

    # K = torch.exp(- ((SS + ZZ))/(2*(sigma**2)))
    # print(res.shape,K.shape)
    return (torch.exp(- ((A+B+C))/(2*(sigma**2))))*res


def kern_met(XX, YY, sigma, sig):
    """Function to compue kernel matrix of Varifolds using JAX

    Args:
        XX (_type_): _description_
        YY (_type_): _description_
        sigma (_type_): _description_
        sig (_type_): _description_

    Returns:
        _type_: _description_
    """
    gamma = 1 / (2*(sigma * sigma))
    gamma_1 = 1/(2*(sig*sig))

    X = XX[:, :k].reshape((1, -1, k))
    Y = YY[:, :k].reshape((1, -1, k))
    u = XX[:, k:][:, None, :]
    v = YY[:, k:][None, :, :]
    # print(u.shape)
    ZZ = torch.sum(X*X, dim=2, keepdim=True) + \
        torch.sum(Y*Y, dim=2, keepdim=True).transpose(2, 1)
    SS = -2*X@Y.transpose(2, 1)

    ss = ((u*v)).sum(-1)

    res = spherical(ss, gamma_1)
    K = torch.exp(- (torch.abs(SS + ZZ))/(2*(sigma**2)))

    return (K*res)


def Err_var(comp, targ, kern_params):
    """Function that computes sq var distance between compressed
       varifold comp, and full varifold of given target shape targ

    Returns:
        _type_: _description_
    """
    targ_extr = var_extract(var_proc(targ))

    sq_norm_1 = E_var_keops_normals(comp, targ_extr, kern_params)

    return sq_norm_1


# 3D varifolds metric
def E_var_keops_normals(v_pars, w_pars, scales):
    """Function to compute Varifolds metric when given
       centres,normals,areas of two functionals (full or compressed)

  Args:
      v_pars (_type_): _description_
      w_pars (_type_): _description_
      scales (_type_): _description_

  Returns:
      _type_: _description_
  """

    v_centres, v_norm_sc, v_weights = v_pars
    w_centres, w_norm_sc, w_weights = w_pars

    sigma_w, sigma_sph = scales

    vv_mat = Var_met(v_centres, v_centres, v_norm_sc,
                     v_norm_sc, sigma_w, sigma_sph)  # sph_zz*h_zz
    ww_mat = Var_met(w_centres, w_centres, w_norm_sc,
                     w_norm_sc, sigma_w, sigma_sph)  # sph_yy*h_yy
    vw_mat = Var_met(v_centres, w_centres, v_norm_sc,
                     w_norm_sc, sigma_w, sigma_sph)
    print(v_weights.shape,w_weights.shape,vw_mat.shape)

    return (v_weights*((vv_mat@v_weights))).sum()  - 2*(v_weights*(vw_mat@w_weights)).sum() + ((ww_mat@w_weights)*w_weights).sum()
