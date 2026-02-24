import jax.numpy as jnp
from pykeops.torch import LazyTensor
from torch.autograd import grad
from .Config import *
import torchode as to
from . import Var_Metrics



def GK(x,y,b,sigma):
    """Keops function to compute kernel matrix, vector reduction

    Args:
        x (_type_): _description_
        y (_type_): _description_
        b (_type_): _description_
        sigma (_type_): _description_

    Returns:
        _type_: _description_
    """
    xx = LazyTensor(x[None,:,:])
    yy = LazyTensor(y[:,None,:])
    bb = LazyTensor(b[:,None,:])

    gamma = 1 / (2*(sigma * sigma))
    D2 = xx.sqdist(yy)
    K = (-D2 * gamma).exp()
    #print(K.shape,bb.shape)
    return (K * bb).sum(0)


def weight_comp_var(idx,GEN,alph,sigma):
  """ Function to compute othogonal projection of varifolds
        given indices

    Args:
        idx (_type_): _description_
        GEN (_type_): _description_
        alph (_type_): _description_
        sigma (_type_): _description_

    Returns:
        _type_: _description_
    """
  gen,normal = GEN
  
  ctrl_pts = gen[idx].contiguous()
  stacked=torch.hstack((ctrl_pts.contiguous(),normal[idx].contiguous() ))
  #spheres = normal[idx].contiguous

  #ctrl_pts = [ctrl,spheres]

  P = ctrl_pts.shape[0]

  K_c = (Var_Metrics.kern_met(stacked.reshape((-1,2*k)),stacked.reshape((-1,2*k) ),*sigma ))


  vals = Var_Metrics.Var_met_reduce(ctrl_pts,gen,normal[idx].contiguous(),normal,alph,*sigma) 

  beta_1 = torch.linalg.solve(K_c + (1e-3)*torch.eye(P).cuda()  ,vals)
  return beta_1#.type(torch.float32)

def H(q, p,scales):
    """Function to compute LDDMM Hamiltonian

    Args: 
        q (_type_): _description_
        p (_type_): _description_
        scales (_type_): _description_

    Returns:
        _type_: _description_
    """
    return Inner(q,q,p,scales)


##rewrite this in torch
def kern(X,Y,scale):
    """Computes kernel matrix in jax

    Args:
        X (_type_): _description_
        Y (_type_): _description_
        scale (_type_): _description_

    Returns:
        _type_: _description_
    """
    ZZ = jnp.sum(X*X,axis=2,keepdims=True) + jnp.sum(Y*Y,axis=2,keepdims=True).transpose([0,2,1])
    SS = -2*X@Y.transpose([0,2,1])
    fin = jnp.exp(-(SS + ZZ)/(2.0*(scale**2)))
    return fin



def HS(q, p,scales):
    """ Calculate and evaluate dynamics function of LDDMM RHS

    Args:
        q (_type_): _description_
        p (_type_): _description_
        scales (_type_): _description_

    Returns:
        _type_: _description_
    """
    potq, potp = grad(H(q, p,scales), (p, q), create_graph=True)
    return potq, -potp



def Inner(x,y,b,scales):
    """Compue kernel matrix vector reduction for Varifolds cross kernel

    Args:
        x (_type_): _description_
        y (_type_): _description_
        b (_type_): _description_
        scales (_type_): _description_

    Returns:
        _type_: _description_
    """
    xx = LazyTensor(x[None,:,:])

    yy = LazyTensor(y[:,None,:])
    bb = LazyTensor(b[:,None,:])

    gamma = 1 / (2*(scales * scales))
    D2 = xx.sqdist(yy)
    K = (-D2 * gamma[0]).exp() + (-D2 * gamma[1]).exp() + (-D2 * gamma[2]).exp() + (-D2 * gamma[3]).exp()
    r = (bb*bb.t())

    return (r*K).sum(0).sum()*.5


def Field(x,y,b,scales):
    """Evaluates a vector field using keops

    Args:
        x (_type_): _description_
        y (_type_): _description_
        b (_type_): _description_
        scales (_type_): _description_

    Returns:
        _type_: _description_
    """
    xx = LazyTensor(x[None,:,:])

    yy = LazyTensor(y[:,None,:])
    bb = LazyTensor(b[:,None,:])

    gamma = 1 / (2*(scales * scales))
    D2 = xx.sqdist(yy)
    K = (-D2 * gamma[0]).exp() + (-D2 * gamma[1]).exp() + (-D2 * gamma[2]).exp() + (-D2 * gamma[3]).exp()
 

    return (K*bb).sum(0) 

