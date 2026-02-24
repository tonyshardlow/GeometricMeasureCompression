import jax
import jax.numpy as jnp
from . import Curr_Metrics
from . import Var_Metrics
from .  import NC_mets
import torch
import numpy as np
from .preproc import loader,var_extract
from .Config import *





def Compressor(params,shape_pars,CTRL,mode):
  """ function that produces weights and indices of compression of measure given control points

  Args:
      params (list): holds spatial and spherical kernel parameters in that order
      delta_pars (list): list of arrays, containing centres and weights respectively
      CTRL (list): _description_
      mode (string): specifies Currents and Varifolds

  Returns:
      array: weights associated to compressins
  """
  
  GEN,alph, = loader(shape_pars,mode)
  ctrl_pts,idx = CTRL
  P = ctrl_pts.shape[0]

  if mode=='Currents':
    print('CURR')
    sigma = params[0]
    K_c = Curr_Metrics.kern_metric(jnp.array(np.array(ctrl_pts.cpu().detach().numpy())).reshape((1,-1,k)),jnp.array(np.array(ctrl_pts.cpu().detach().numpy())).reshape((1,-1,k)),sigma)[0]
    K_c = jnp.array(K_c,dtype=jnp.float64)

    vals =  Curr_Metrics.GK(ctrl_pts.reshape((-1,k)),GEN.reshape((-1,k)),alph,sigma).cpu().detach().numpy()
  

    beta_1 = jnp.linalg.solve(K_c + 1e-12*jnp.eye(P),vals) 

  if mode == 'Varifolds':
    print("VAR")
    sigma,sigma_sph = params
    v_areas = alph
    
    K_c = Var_Metrics.kern_m(jnp.array(np.array(ctrl_pts.cpu().detach().numpy())),jnp.array(np.array(ctrl_pts.cpu().detach().numpy())),sigma,sigma_sph)
  
    K_c = jnp.array(K_c,dtype=jnp.float32)

    x,y,u,v = ctrl_pts[:,:k],GEN[:,:k],ctrl_pts[:,k:],GEN[:,k:]

    vals = Var_Metrics.Var_met_reduce(x,y,u,v,v_areas,sigma,sigma_sph).cpu().detach().numpy()

    beta_1 = jnp.linalg.solve(K_c + (1e-12)*np.eye(P) ,vals) 

  elif mode=='NC':

    print("NC MODE")
    sigma = params[0]
    K_c =  Curr_Metrics.kern_metric(jnp.array(np.array(ctrl_pts.cpu().detach().numpy())).reshape((1,-1,k)),jnp.array(np.array(ctrl_pts.cpu().detach().numpy())).reshape((1,-1,k)),sigma)[0]
    #NC_mets.kern_metric(ctrl_pts,ctrl_pts,sigma)
    K_c = np.array(K_c,dtype=np.float64)

    vals = NC_mets.GK(ctrl_pts.reshape((-1,k)),GEN.reshape((-1,k)),alph,sigma).cpu().detach().numpy()

    beta_1 = np.linalg.solve(K_c + 1e-12*np.eye(ctrl_pts.shape[0]),vals) 

  return beta_1
   

  #return idx,beta_1



def orthog_pursuit(params,shape_pars,P):
  """ Compressor for currents using orthogonal matching pursuit algorithm

  Args:
      params (double): _description_
      shape_pars (_type_): _description_
      P (_type_): _description_

  Returns:
      _type_: _description_
  """
  gen,alph = loader(shape_pars,'Currents')
  n = gen.shape[0]
  
  ids=[]
  pts=[]
  errs = []
  p=0

  initial= Curr_Metrics.GaussKernel(gen.reshape((-1,k)).cuda(),gen.reshape((-1,k)).cuda(),alph,*params)
  res = initial 

  while p<P:

    print(p)
    p +=1

    norms = torch.linalg.norm(res,dim=1)
    ls = list(norms)
    index = ls.index(max(ls))
    ids.append(index)
    pts.append(np.array(gen[index].cpu().detach().numpy()))
   

    KK = torch.tensor(np.array(Curr_Metrics.kern_metric(np.array(pts).reshape((1,-1,k)),np.array(pts).reshape((1,-1,k)),*params)[0])).cuda() 
 
    beta = torch.linalg.solve(KK ,initial[np.array(ids)] ).cuda()
    moment = torch.zeros((n,k),device=device)
  
    moment[np.array(ids)] = beta

    res = initial - Curr_Metrics.GaussKernel(gen.reshape((-1,k)).cuda(),torch.tensor(np.array(pts).reshape((-1,k))).cuda(),beta,*params)#.cuda()#.cpu().detach().numpy()
   

  return ids,beta




def var_obj(par_opt,target,scales):
  """Objective for compression of Varifolds with optimization
    computes var distances sq

  Args:
      par_opt (_type_): _description_
      target (_type_): _description_
      scales (_type_): _description_

  Returns:
      _type_: _description_
  """
  p_opt = [par_opt[0],par_opt[1]]
  #form unit vec and area weights
  opt_pars=var_extract(p_opt)
  targ_pars=var_extract(target)

  ##compute metric here
  var_dist = Var_Metrics.E_var_keops_normals(opt_pars,targ_pars,scales)

  return var_dist


def var_quantizer(par_opt,target,scales,num_iter,step):
  """Compressor for Varifolds using optimizationm returns optimal parameters

  Args:
      par_opt (_type_): _description_
      target (_type_): _description_
      scales (_type_): _description_
      num_iter (_type_): _description_
      step (_type_): _description_

  Returns:
      _type_: _description_
  """
  par_opt.requires_grad=True
  optimizer = torch.optim.LBFGS([par_opt],lr=step)

  def closure():
      
      optimizer.zero_grad()
      L = var_obj(par_opt,target,scales)
      print("loss", L.item())

      L.backward()
      L.detach()
      return L

  for i in range(num_iter):
    optimizer.step(closure)


  return par_opt

