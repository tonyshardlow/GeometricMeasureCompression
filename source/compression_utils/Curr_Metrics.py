#%%
import jax
import jax.numpy as jnp
import pykeops
from pykeops.torch import LazyTensor
from .Config import*
#%%
#%%
############################# REFACTOR
@jax.jit
def kern_met(X,Y,scale):
  """Function to compute kernel matrix of Current in JAX

  Args:
      X (_type_): _description_
      Y (_type_): _description_
      scale (_type_): _description_

  Returns:
      _type_: _description_
  """
  X=X.reshape((1,-1,k))
  Y=Y.reshape((1,-1,k))
  ZZ = jnp.sum(X*X,axis=2,keepdims=True) + jnp.sum(Y*Y,axis=2,keepdims=True).transpose([0,2,1])
  SS = -2*X@Y.transpose([0,2,1])
  fin = jnp.exp(- ( jnp.abs(SS + ZZ) )/(2*(scale**2) ) )
  return fin[0]


@jax.jit
def kern_metric(X,Y,scale):
  """ Function to compute kernel matrix in JAX returns different shape

  Args:
      X (_type_): _description_
      Y (_type_): _description_
      scale (_type_): _description_

  Returns:
      _type_: _description_
  """
  ZZ = jnp.sum(X*X,axis=2,keepdims=True) + jnp.sum(Y*Y,axis=2,keepdims=True).transpose([0,2,1])
  SS = -2*X@Y.transpose([0,2,1])
  fin = jnp.exp(- ( jnp.abs(SS + ZZ) )/(2*(scale**2) ) )
  return fin

############################# REFACTOR

def GaussKernel(x,y,b,sigma):
    """ Function to compute dual vector field for Currents using KEOPS

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
 
    return (K * bb).sum_reduction(axis=0)

def GK(x,y,b,sigma):
    """ Function to compute dual vector field for Currents (SAME AS PREV FUNC) 

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

    return (K * bb).sum(0)


def Err_curr(comp_pars,targ_pars,kern_params):
  """ Function to compute sq Currents metrics with delta inputs

  Args:
      comp_pars (_type_): _description_
      targ_pars (_type_): _description_
      kern_params (_type_): _description_

  Returns:
      _type_: _description_
  """
  idx,beta_1 = comp_pars
  gen,alph =  targ_pars
  sigma = kern_params

  moment1 = jnp.zeros((n,k))
  moment1 = moment1.at[idx].set(beta_1)

  residual_1=torch.tensor(np.array(moment1)).cuda().type(torch.float64)-alph

  print("norm computation...")
  sq_norm_1 = sq_norm_1 = (((residual_1*GaussKernel(gen.cuda().reshape((-1,k)),gen.cuda().reshape((-1,k)),residual_1,sigma ) ).sum())) 

  return sq_norm_1