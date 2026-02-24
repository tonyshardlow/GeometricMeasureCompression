import jax.numpy as jnp
from pykeops.torch import LazyTensor
from torch.autograd import grad
from .Config import *
import torchode as to
from .Var_Metrics import E_var_keops_normals
from .preproc import var_proc,var_extract
from . import Hamiltonian as Ham
from .Hamiltonian import weight_comp_var
from . import NC_mets,Curr_Metrics



class Geodesic_shooting():
  """A class 
        for fitting matching geodesic diffeomorphisms, using momentum conservation.
    """

  def __init__(self, 
               temp,targ,spatial_params,p_init,metric,comp):
    """Initiatalisation requires
       template, target (in delta form), space kernel params, initial momentum guess,
       metric and metric params

    Args:
        temp (_type_): _description_
        targ (_type_): _description_
        spatial_params (_type_): _description_
        p_init (_type_): _description_
        metric (_type_): _description_

    Returns:
        _type_: _description_
    """
   
    self.gen_pts,self.M1,self.idx,self.ctrl_idx = temp
    
      #self.gen_pts,self.M1,self.idx,self.ctrl_idx = temp

    self.n = self.gen_pts.shape[0]
    self.targ_pars = targ
    self.comp = comp

    self.P = self.ctrl_idx.shape[0]
    self.k = gen_pts.shape[1]
  
    self.metric,self.sigma = metric
    self.scales = spatial_params
    
    self.p_init = p_init

    def __repr__(self):
      return repr("A class for fitting matching geodesic diffeomorphisms, using momentum conservation")

    def f(t,y):
   
      q,p=torch.split(y.reshape((self.n+self.P,k)),[self.n,self.P],0)
      
      res1,res2 = Ham.HS(q[self.ctrl_idx],p,self.scales)
      res = Ham.Field(q,q[self.ctrl_idx],p,self.scales) 
      result = torch.cat((res,res2))
      return result.reshape((1,-1))



    n_steps = 3
    self.t_eval = torch.linspace(0.0, 1.0, n_steps,device=device)

    term = to.ODETerm(f)
    step_method = to.Euler(term=term)
    step_size_controller =to.FixedStepController()
    self.adjoint = to.AutoDiffAdjoint(step_method, step_size_controller)



  def geodesic(self,p):
    y0 = torch.cat((self.gen_pts,p ))
    problem = to.InitialValueProblem(y0=y0.reshape((1,-1)), t_eval=self.t_eval.reshape((1,-1)))
    sol = self.adjoint.solve(problem,dt0= torch.tensor(.1).cuda())
    return torch.split(sol.ys[:,-1,:].reshape((self.n+self.P,k)),[self.n,self.P],0)[0]



  def output(self):
    return self.geodesic(self.p_init).cpu().detach().numpy()



  def optimize(self,iters):

    #optimizer = torch.optim.Adam([self.p_init],
       #                           lr=1e-4,
                                 
      #                            )
    optimizer = torch.optim.LBFGS([self.p_init],
                                  line_search_fn='strong_wolfe'
                                 
                                  )

    def closure():

      optimizer.zero_grad()
      L = self.objective(self.p_init)
      print("loss", L.item()," ",self.P)

      L.backward()
      L.detach()
      return L
    for i in range(iters):
      optimizer.step(closure)

    return None

  def objective(self,p_init):
      final_q = self.geodesic(p_init)      
      
      q_int = self.gen_pts[self.ctrl_idx]

      
      energy = Ham.H(q_int,p_init,self.scales)
      #METRIC PART CAN BE CHANGED!!!
      if self.metric=='Varifolds':
        final = var_extract(var_proc([final_q,M1]))
        if self.comp:
        
          beta=weight_comp_var(self.idx,[final[0],final[1]],final[2],self.sigma).cuda()

          ##need to index this by idx 
          q_par = [final[0][self.idx],final[1][self.idx],beta.reshape((self.idx.shape[0]))]

        else:
          q_par = [final[0],final[1],final[2] ]

        res=E_var_keops_normals(q_par,self.targ_pars,self.sigma)

      if self.metric=='NC':
        pt1,w2 = self.targ_pars
        fin_parts=  NC_mets.get_parts(final_q,M1,shape_pars[-1],*shape_pars[2])
        fin_cent,fin_weight = NC_mets.Compute_weights_centres(fin_parts,shape_pars[3])
        if self.comp: 
          pt,w1 = NC_mets.Compress_NC(fin_cent,fin_weight,self.idx,NC_mets.kern_metric,NC_mets.GK,self.sigma)
       
        else:
          pt,w1 = fin_cent,fin_weight
          
        res = NC_mets.NC_d_comp(pt.contiguous(),pt1.contiguous(),w1.contiguous(),w2.contiguous(),self.sigma)
        

      return  100*res + energy