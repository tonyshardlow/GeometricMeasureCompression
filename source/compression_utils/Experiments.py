from .DAC import DAC,DAC_vec
from .RLS import recursiveNystrom
import jax.numpy as jnp
from .Compressors import Compressor
from .preproc import loader
from .Config import *
from . import NC_mets
from . import Curr_Metrics
from . import Var_Metrics


def Experiment(exp_pars,shape_pars):
  """ Function to perform and return [weights,control points,idx], using experiemnt with parameters
      exp_pars, and shape parameters shape_pars
  Args:
      exp_pars (_type_): _description_
      shape_pars (_type_): _description_

  Returns:
      _type_: _description_
  """

  Ps,trials,sampler,mode,kern_params= exp_pars

  
  #if mode=='Currents':
  #  kern_met = Curr_Metrics.kern_met
  if mode=='Currents':
    if sampler=='DAC':
      kern_met =  NC_mets.kern_met_torch 
    else:
      kern_met = Curr_Metrics.kern_met

  if mode=='Varifolds':
    kern_met = Var_Metrics.kern_met_new
  
    
  if mode=='NC':
    if sampler=='DAC':
      kern_met =  NC_mets.kern_met_torch  # NC_mets.kern_metric
    if sampler=='RLS':
      kern_met =  NC_mets.kern_met

  betas = []
  ctrls = []
  delta_pars = loader(shape_pars,mode)
  GEN = delta_pars[0]

  n = GEN.shape[0]
  
  if sampler=='DAC':
    scores=DAC_vec(GEN, .1, 100, kern_met, kernel_param)
  
  print("original resolution: ",n," triangles")
  if sampler=='uniform':
    scores = np.ones((GEN.shape[0]))
 
  for P in Ps:
    print("compressing to size: ",P)

    for i in range(trials):

      if sampler=='RLS':
        _,scores = recursiveNystrom(GEN.cpu().detach().numpy(), P, kern_met, kernel_param,accelerated_flag=True, random_state=None, lmbda_0=0, return_leverage_score=True)

      idx=np.random.choice(n,P,replace=False,p=jnp.array(scores/scores.sum(),dtype=jnp.float64))
     
      ctrl_pts = delta_pars[0][np.array(idx)]
      CTRL = [ctrl_pts,idx]

      
      beta_1 =  Compressor(kern_params,shape_pars,CTRL,mode) 

      betas.append(beta_1)
      ctrls.append(CTRL)
      
  return betas,ctrls#,idx
  