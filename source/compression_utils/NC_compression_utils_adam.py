#%%

import pykeops
import jax
import torch
from pykeops.torch import LazyTensor,Genred,Vi,Vj,Pm
from pykeops.numpy import LazyTensor as LazyTensor_np
import numpy as np
import jax.numpy as jnp


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## normalization function
def normalize(shape):
  return (shape - shape.mean(0))/shape.std(0).mean()


def pre_comp_nc_vec_con(gen_pts,M1,ex_bdry):
  """takes in shape, mesh, bdry boolean and extracts
      required info for downline tasks with Normal Cycles.

  Args:
      gen_pts (array): vertices of input shape of size (n,3)
      M1 (array): mesh structure of input shape of size (m,3)
      ex_bdry (bool): True if input shape has boundary, false otherwise.

  Returns:
     list : a list of useful structural indices useful for NC metric.

  """
  shapes = []

  max_ = 0#

  edges_for_verts_inds = []

  #########

  #for each unique edge compute traingles attached
  Acc = [[0,1],[2,0],[1,2]]
  Rej = [[1,0],[0,2],[2,1]]

  e1 = torch.vstack([M1[:,0],M1[:,1]]).T
  e2 = torch.vstack([M1[:,1],M1[:,2]]).T
  e3 = torch.vstack([M1[:,2],M1[:,0]]).T
  #indices of unique edges
  u_edge_inds = torch.unique(torch.vstack([e1,e2,e3]),dim=0)#np.unique(torch.vstack([e1,e2,e3]).cpu().detach().numpy(),axis=0)
  u_edge_inds = torch.unique(torch.sort(u_edge_inds,axis=1)[0],dim=0)


  VS = []
  nrms = []
  lens = []

  from collections import defaultdict

  dic = defaultdict(list)

  print("looping on triangulation")

  for i in range(len(M1)):
    e_1 = tuple(sorted((M1[i,0].item(),M1[i,1].item())))#.sort()
    e_2 = tuple(sorted((M1[i,1].item(),M1[i,2].item())))#.sort()
    e_3 = tuple(sorted((M1[i,2].item(),M1[i,0].item())))#.sort()

    dic[e_1].append(i)
    dic[e_2].append(i)
    dic[e_3].append(i)

  #print(dic.keys(),dic.values())
  #store boundary edges in this
  bdry = []
  print('collecting bdry edges')
  print(u_edge_inds.shape[0])
  for inds in range(u_edge_inds.shape[0]):
 
    vs = dic[tuple(sorted((u_edge_inds[inds][0].item(), u_edge_inds[inds][1].item() ))) ]
    #print(vs)

    if len(vs ) ==1:
      bdry.append(u_edge_inds[inds] )
      VS.append(np.array((vs[0],-1,-1,-1)) )
      lens.append(True)
    else:
      #print(np.where(vs)[0])
      VS.append(np.append(np.array(vs),[-1]*(4-len(vs )) ) )
      lens.append(False)


  stack=np.vstack(VS).astype(int)

  if ex_bdry:
    bdry_verts = torch.unique(torch.vstack(bdry).reshape((-1,1)))
    bdry_stack = torch.vstack(bdry)
    lis = [ np.where((bdry_stack == bdry_verts[i]).sum(1))[0].shape for i in range(len(bdry_verts ))]
    max_lis = max(lis)
    min_lis=min(lis)

    if max_lis==min_lis:
      bdry_vert_ed_inds = np.vstack([ np.where((bdry_stack == bdry_verts[i]).sum(1))[0] for i in range(len(bdry_verts ))])
      bdry_vert_edges = bdry_stack[bdry_vert_ed_inds]

    else:
      bdry_vert_ed_inds = np.vstack([ np.append(np.where((bdry_stack == bdry_verts[i]).sum(1))[0],[0]*(max_lis[0]-lis[i][0] )) for i in range(len(bdry_verts ))])
      bdry_vert_edges = bdry_stack[bdry_vert_ed_inds]
      #print(bdry_vert_edges[10,2:,:])
      for i in range(len(bdry_vert_edges)):
        print(int(max_lis[0]-lis[i][0]))
        bdry_vert_edges[i,int(max_lis[0]-lis[i][0]):,:] = 0.0
    print(bdry_vert_edges)
  else:
    bdry_verts = None#torch.unique(torch.vstack(bdry).reshape((-1,1)))
    bdry_stack = None#torch.vstack(bdry)

    bdry_vert_ed_inds = None#np.vstack([ np.where((bdry_stack == bdry_verts[i]).sum(1)) for i in range(len(bdry_verts ))])
    bdry_vert_edges = None#bdry_stack[bdry_vert_ed_inds]

  coords = coords = [ torch.zeros((stack.shape[0],1)) for i in range(4)] # [coord1,coord2]

  for index,edge in enumerate(u_edge_inds):

    for i in range(4):
      #print(edge)
      if stack[index][i]==-1:
        continue

      test=[list(M1[stack[index][i]]).index(int(edge[0])) ,list(M1[stack[index][i]]).index(int(edge[1]))]

      if test in Acc:
        coords[i][index] = 1.0#print("Acc")
      else:
        coords[i][index] = -1.

  return stack,edges_for_verts_inds,u_edge_inds,max_,lens,coords,bdry_verts,bdry_vert_edges



def get_parts(gen_pts,M1,bdry,*args):

  """Takes in the shape, mesh, bdry info and args consisting of 
     shape structure info

  Returns:
     list: list of quantities requires to compute NC metric:
     fs
     boundary verts
     centres of edges
     normals
     sum_edges
  """

  stack,edges_for_verts_inds,u_edge_inds,max_,lens,coords,bdry_verts,bdry_vert_edges = args
  if bdry:
    e_for_verts = (gen_pts[bdry_vert_edges[:,:,1] ] - gen_pts[bdry_vert_edges[:,:,0] ] )#.unsqueeze(1)

    f_norm = torch.linalg.norm(e_for_verts,dim=2) #+ .000001

    f_norm_1 = torch.where(f_norm != 0.0,f_norm,np.inf)
    #print(f_norm)
    sev_vec = e_for_verts/f_norm_1.unsqueeze(-1)
    sev  = sev_vec.sum(1)
    sum_edges = sev

  zz = gen_pts
  v1,v2,v3= zz[M1[:,0]].contiguous(),zz[M1[:,1]].contiguous(),zz[M1[:,2]].contiguous()
  g1 = v2-v1
  g2 = v3-v1
  g3 = v3-v2
  v_norms = .5*torch.cross(g1,g2)

  v_normals = v_norms/torch.linalg.norm(v_norms,dim=1).reshape((-1,1))
  v_normal_mod = torch.vstack([v_normals,torch.tensor([0.0,0.0,0.0]).cuda() ])

  v_re = torch.zeros(stack.shape[0],3).cuda()
  for i in range(4):
    v_re += coords[i].cuda()*v_normal_mod[stack[:,i]].cuda()  #+ coord2*v_normals[stack[:,1]]

  v_res=v_re

  fs = (gen_pts[u_edge_inds][:,1,:].contiguous() -  gen_pts[u_edge_inds][:,0,:].contiguous())
  cs =  (gen_pts[u_edge_inds][:,1,:].contiguous() +  gen_pts[u_edge_inds][:,0,:].contiguous())/2
  normals = v_res

  #print(sev.shape,sev_vec.shape)
  if bdry:
    return fs.contiguous(),gen_pts[bdry_verts].contiguous(),cs.contiguous(),normals.contiguous(),sum_edges.contiguous()#[bdry_verts]
  else:
    return fs.contiguous(),None,cs.contiguous(),normals.contiguous(),None
  

#def comm(a,b,i,j):
 # return (a[:,i-1]*b[:,j-1]) - (a[:,j-1]*b[:,i-1])

indexes = [[i,j] for i in range(1,6) for j in range(i+1,7)  ]
#indexes_n = [[1,2],[1,3],[1,4],[1,5],[1,6],[2,3],[2,4],[2,5],[2,6],[3,4],[3,5],[3,6],[4,5],[4,6],[5,6]]

indexes_1 = [indexes[i][0]-1 for i in range(len(indexes))]
indexes_2 = [indexes[i][1]-1 for i in range(len(indexes))]

#indexes_1_n = [0,0,0,0,0,1,1,1,1,2,2,2,3,3,4]
#indexes_2_n = [1,2,3,4,5,2,3,4,5,3,4,5,4,5,5]


#used to compute wedge product normal cycle embedding
def coordinates(a,b):

  return (a[:,indexes_1]*b[:,indexes_2] - a[:,indexes_2]*b[:,indexes_1]).reshape((-1,15)).cuda()


#takes in shape structure and computes weights of dirac deltas
def Embed(S1):

  #compute weights
  a1=torch.zeros((S1[0].shape[0],6))
  norms_ = torch.linalg.norm(S1[0],dim=1).reshape((-1,1))
  normalized= S1[0]/norms_
  a1[:,:3] = normalized
  a2 =torch.zeros((S1[0].shape[0],6))
  sum_norm=S1[-2]
  a2[:,3:] = sum_norm
  #fill is the boundary weights

  if S1[1] is not None:

    fill=torch.zeros((S1[1].shape[0],15)).cuda()
    fill[:,12:] = S1[-1]
    return torch.vstack([fill,norms_*coordinates(a1,a2)])

  else:
    return norms_*coordinates(a1,a2)


#convenience function for evaluating dual function
def GK(x,y,b,sigma):
  
    xx = LazyTensor(x[None,:,:])
    yy = LazyTensor(y[:,None,:])
    bb = LazyTensor(b[:,None,:])

    gamma = 1 / (2*(sigma * sigma))
    D2 = xx.sqdist(yy)
    K = (-D2 * gamma).exp()
    #print(K.shape,bb.shape)
    return (K * bb).sum(0)

#gaussian kernel function
def kern(X,Y,sigma_w):
 
  ZZ = torch.sum(X*X,axis=1,keepdims=True) + torch.sum(Y*Y,axis=1,keepdims=True).transpose(1,0)
  SS = -2*X@Y.transpose(1,0)
  fin = torch.exp(-(SS + ZZ)/(2.0*(sigma_w**2)))
  return fin

#computes gaussian kernel function
def kern_met_NC(X,Y,scale):
 
  k=3
  #X=X.reshape((1,-1,k))
  #Y=Y.reshape((1,-1,k))
  ZZ = torch.sum(X*X,axis=2,keepdim=True) + torch.sum(Y*Y,axis=2,keepdim=True).transpose(2,1 )
  SS = -2*X@Y.transpose(2,1)
  fin = torch.exp(- ( torch.abs(SS + ZZ) )/(2*(scale**2) ) )
  return fin


#performs orthogonal projection 
def weight_comp(idx,GEN,alph,sigma):
  """_summary_

  Args:
      idx (_type_): _description_
      GEN (_type_): _description_
      alph (_type_): _description_
      sigma (_type_): _description_

  Returns:
      _type_: _description_
  """

  k=3



  ctrl_pts = GEN[idx].contiguous()

  P = ctrl_pts.shape[0]
  K_c = kern(ctrl_pts.reshape((-1,k)),ctrl_pts.reshape((-1,k) ),sigma.item())#.type(torch.float64)#[0]


  vals = GK(ctrl_pts.reshape((-1,k)),GEN.reshape((-1,k)),alph,sigma.item())#.cpu().detach().numpy()

  beta_1 = torch.linalg.solve(K_c + 1e-3*torch.eye(P,device=device).type(torch.float64),vals.type(torch.float64))
  return beta_1.type(torch.float32)

#computes RLS sampling scores given shape X, kernel and scale
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
#%%
#computes normal cycle inner product
def N_compressed(pt,pt1,w1,w2,sigma):
   
    gamma = 1 / (2*(sigma * sigma))
    #gamma_1 = 1 / (2*(sigma_sph * sigma_sph))
    #x, y, b = Vi(0, 3), Vj(1, 3), Vj(2, 3)
    pt = LazyTensor(pt[:,None,:])
    pt1 = LazyTensor(pt1[None,:,:])

    D = pt.sqdist(pt1)
    K = (-D * gamma).exp()

    w1 = LazyTensor(w1[:,None,:])
    w2 = LazyTensor(w2[None,:,:])

    ss = ((w1*w2) ).sum()
    res = K*ss

    return res.sum(0).sum()

#computes normal cycles metric given delta centres and weights
def NC_metric(pts,w,pts1,w1,sigma):

    return N_compressed(pts,pts,w,w,sigma) - (2*N_compressed(pts,pts1,w,w1,sigma) ) + N_compressed(pts1,pts1,w1,w1,sigma)
