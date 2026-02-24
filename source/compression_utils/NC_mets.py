import torch
import numpy as np
import jax.numpy as jnp
from pyntcloud import PyntCloud
from jax import config
config.update("jax_enable_x64", True)
import pywavefront
import stripy
import pykeops
import jax
from pykeops.torch import LazyTensor

def pre_comp_nc_vec_con(gen_pts,M1,ex_bdry):

  shapes = []


  #for index in range(gen_pts.shape[0]):
  #  A = list(np.unique(np.array(M1[np.where((M1==index).sum(1))])))
  #  if index in A:
      #print(index)
  #    A.remove(index)
  #  shapes.append(len(A))

  max_ = 0# max(shapes)
  #print(max_)

  edges_for_verts_inds = []

  #for index in range(gen_pts.shape[0]):
  #  A = list(np.unique(np.array(M1[np.where((M1==index).sum(1))] ) ))
  #  if index in A:
      #print(index)
  #    A.remove(index)
  #  A = A + (max_-len(A))*[index]

  #  edges_for_verts_inds.append(A)

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
    #vs=( ( (M1==u_edge_inds[inds][0]).sum(1)  + (M1==u_edge_inds[inds][1]).sum(1)) ==2)
    #print(tuple(sorted(u_edge_inds[inds])))
    vs = dic[tuple(sorted((u_edge_inds[inds][0].item(), u_edge_inds[inds][1].item() ))) ] 
    #print(vs)

    #if len(np.where(vs)[0] ) ==1:
    #  bdry.append(u_edge_inds[inds])
    #  VS.append((np.array((np.where(vs)[0][0],np.where(vs)[0][0])),) )
    #  lens.append(True)
    if len(vs ) ==1:
      bdry.append(u_edge_inds[inds] )
      VS.append(np.array((vs[0],-1,-1,-1)) ) 
      lens.append(True)
    else:
      #print(np.where(vs)[0])
      VS.append(np.append(np.array(vs),[-1]*(4-len(vs )) ) )
      lens.append(False)

  #store boundary edges in this
 # bdry = []


  #for inds in range(u_edge_inds.shape[0]):
  #  vs=( ( (M1==u_edge_inds[inds][0]).sum(1)  + (M1==u_edge_inds[inds][1]).sum(1)) ==2)
  
    #print(inds)
    #print(np.where(vs)[0].shape)
  #  if len(np.where(vs)[0] ) ==1:
      #print(np.where(vs)[0] ,u_edge_inds[inds])
  #    bdry.append(u_edge_inds[inds])
  #    VS.append((np.array((np.where(vs)[0][0],-1,-1,-1 ) ),) )
  #    lens.append(True)
  #  else:
      #print(np.where(vs)[0])
      #print(len())
      #VS.append(np.hstack([np.where(vs),-1,-1,-1 ]))
   #   VS.append(np.append(np.where(vs)[0],[-1]*(4-len(np.where(vs)[0])) ))
   #   lens.append(False)

  stack=np.vstack(VS).astype(int)

  #print(len(bdry),bdry)
  #get unique indices of boundary vertices
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
  #print(bdry_vert_edges.shape),#[ np.where((bdry_stack == bdry_verts[i]).sum(1)) for i in range(len(bdry_verts ))])
  #print(torch.vstack(bdry).shape,bdry_verts.shape)
  #print(VS[0].shape)
  #stack = np.vstack(VS)


  #coord1 = torch.ones((stack.shape[0],1))
  #coord2 = torch.ones((stack.shape[0],1))
  coords = coords = [ torch.zeros((stack.shape[0],1)) for i in range(4)] # [coord1,coord2]

  for index,edge in enumerate(u_edge_inds):
#print(MM[stack[index]],edge)
    #print(index,edge)
    for i in range(4):
      #print(edge)
      if stack[index][i]==-1:
        continue

      test=[list(M1[stack[index][i]]).index(int(edge[0])) ,list(M1[stack[index][i]]).index(int(edge[1]))]
    #01 g 10 b 02 b 20 g 12 g 21 b
      #if index==-1:
      #  coords[i][index] = 0.0
      if test in Acc:
        coords[i][index] = 1.0#print("Acc")
      else:
        coords[i][index] = -1.
  
  #for index,edge in enumerate(u_edge_inds):
  #print(MM[stack[index]],edge)
   # for i in range(2):
    #  test=[list(M1[stack[index][i]]).index(edge[0]) ,list(M1[stack[index][i]]).index(edge[1])]
    #01 g 10 b 02 b 20 g 12 g 21 b
    #  if test in Acc:
     #   coords[i][index] = 1.0#print("Acc")
     # else:
      #  coords[i][index] = -1.
      #print(test)

  #get boundary vertices indices
  return stack,edges_for_verts_inds,u_edge_inds,max_,lens,coords,bdry_verts,bdry_vert_edges



def get_parts(gen_pts,M1,bdry,*args):

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

  #coord1 = coords[0].cuda()
  #coord2 = coords[1].cuda()
  #print(coord1)

  #v_re = coord1*v_normals[stack[:,0]] + coord2*v_normals[stack[:,1]]

  #ints = torch.ones(len(lens)).cuda()
  #ints[lens] = 0.5
  #print(ints.mean())
  #v_res = ints.reshape((-1,1))*v_re

  v_normal_mod = torch.vstack([v_normals,torch.tensor([0.0,0.0,0.0]).cuda() ])

  v_re = torch.zeros(stack.shape[0],3).cuda()
  for i in range(4):
    v_re += coords[i].cuda()*v_normal_mod[stack[:,i]].cuda()  #+ coord2*v_normals[stack[:,1]]

  #ints = torch.ones(len(lens)).cuda()
  #ints[lens] = 0.5
  #print(ints.mean())
  #v_res = ints.reshape((-1,1))*v_re
  v_res=v_re
  #if sum(lens)>0:
   # v_res[lens] = v_res[lens]/2.0
  fs = (gen_pts[u_edge_inds][:,1,:].contiguous() -  gen_pts[u_edge_inds][:,0,:].contiguous())
  cs =  (gen_pts[u_edge_inds][:,1,:].contiguous() +  gen_pts[u_edge_inds][:,0,:].contiguous())/2
  normals = v_res
  
  #print(sev.shape,sev_vec.shape)
  if bdry:
    return fs.contiguous(),gen_pts[bdry_verts].contiguous(),cs.contiguous(),normals.contiguous(),sum_edges.contiguous()#[bdry_verts]
  else:
    return fs.contiguous(),None,cs.contiguous(),normals.contiguous(),None
  
def pre_comp_nc_vec_con_old(gen_pts,M1,ex_bdry):

  shapes = []


  for index in range(gen_pts.shape[0]):
    A = list(np.unique(np.array(M1[np.where((M1==index).sum(1))])))
    if index in A:
      #print(index)
      A.remove(index)
    shapes.append(len(A))

  max_ = max(shapes)
  print(max_)

  edges_for_verts_inds = []

  for index in range(gen_pts.shape[0]):
    A = list(np.unique(np.array(M1[np.where((M1==index).sum(1))] ) ))
    if index in A:
      #print(index)
      A.remove(index)
    A = A + (max_-len(A))*[index]

    edges_for_verts_inds.append(A)

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

  #store boundary edges in this
  bdry = []

  for inds in range(u_edge_inds.shape[0]):
    vs=( ( (M1==u_edge_inds[inds][0]).sum(1)  + (M1==u_edge_inds[inds][1]).sum(1)) ==2)
    if len(np.where(vs)[0] ) ==1:
      bdry.append(u_edge_inds[inds])
      VS.append((np.array((np.where(vs)[0][0],np.where(vs)[0][0])),) )
      lens.append(True)
    else:
      #print(np.where(vs)[0])
      VS.append(np.where(vs))
      lens.append(False)

  #print(len(bdry),bdry)
  #get unique indices of boundary vertices
  if ex_bdry:
    bdry_verts = torch.unique(torch.vstack(bdry).reshape((-1,1)))
    bdry_stack = torch.vstack(bdry)

    bdry_vert_ed_inds = np.vstack([ np.where((bdry_stack == bdry_verts[i]).sum(1)) for i in range(len(bdry_verts ))])
    bdry_vert_edges = bdry_stack[bdry_vert_ed_inds]
  
  else:
    bdry_verts = None#torch.unique(torch.vstack(bdry).reshape((-1,1)))
    bdry_stack = None#torch.vstack(bdry)

    bdry_vert_ed_inds = None#np.vstack([ np.where((bdry_stack == bdry_verts[i]).sum(1)) for i in range(len(bdry_verts ))])
    bdry_vert_edges = None#bdry_stack[bdry_vert_ed_inds]
  #print(bdry_vert_edges.shape),#[ np.where((bdry_stack == bdry_verts[i]).sum(1)) for i in range(len(bdry_verts ))])
  #print(torch.vstack(bdry).shape,bdry_verts.shape)
  #print(VS[0].shape)
  stack = np.vstack(VS)


  coord1 = torch.ones((stack.shape[0],1))
  coord2 = torch.ones((stack.shape[0],1))
  coords = [coord1,coord2]

  for index,edge in enumerate(u_edge_inds):
  #print(MM[stack[index]],edge)
    for i in range(2):
      test=[list(M1[stack[index][i]]).index(edge[0]) ,list(M1[stack[index][i]]).index(edge[1])]
    #01 g 10 b 02 b 20 g 12 g 21 b
      if test in Acc:
        coords[i][index] = 1.0#print("Acc")
      else:
        coords[i][index] = -1.
      #print(test)

  #get boundary vertices indices
  return stack,edges_for_verts_inds,u_edge_inds,max_,lens,coords,bdry_verts,bdry_vert_edges


def get_parts_old(gen_pts,M1,bdry,*args):

  stack,edges_for_verts_inds,u_edge_inds,max_,lens,coords,bdry_verts,bdry_vert_edges = args
  
  
  if bdry:
    e_for_verts = (gen_pts[bdry_vert_edges[:,:,1] ] - gen_pts[bdry_vert_edges[:,:,0] ] )#.unsqueeze(1)

    f_norm = torch.linalg.norm(e_for_verts,dim=2) + .000001

    sev_vec = e_for_verts/f_norm.unsqueeze(-1) 
    sev  = sev_vec.sum(1)
    sum_edges = sev

  zz = gen_pts
  v1,v2,v3= zz[M1[:,0]].contiguous(),zz[M1[:,1]].contiguous(),zz[M1[:,2]].contiguous()
  g1 = v2-v1
  g2 = v3-v1
  g3 = v3-v2
  v_norms = .5*torch.cross(g1,g2)

  v_normals = v_norms/torch.linalg.norm(v_norms,dim=1).reshape((-1,1)) 

  coord1 = coords[0].cuda()
  coord2 = coords[1].cuda()
  #print(coord1)

  v_re = coord1*v_normals[stack[:,0]] + coord2*v_normals[stack[:,1]]

  ints = torch.ones(len(lens)).cuda()
  ints[lens] = 0.5
  #print(ints.mean())
  v_res = ints.reshape((-1,1))*v_re
  #if sum(lens)>0:
   # v_res[lens] = v_res[lens]/2.0
  fs = (gen_pts[u_edge_inds][:,1,:].contiguous() -  gen_pts[u_edge_inds][:,0,:].contiguous())
  cs =  (gen_pts[u_edge_inds][:,1,:].contiguous() +  gen_pts[u_edge_inds][:,0,:].contiguous())/2
  normals = v_res
  
  #print(sev.shape,sev_vec.shape)
  if bdry:
    return fs.contiguous(),gen_pts[bdry_verts].contiguous(),cs.contiguous(),normals.contiguous(),sum_edges.contiguous()#[bdry_verts]
  else:
    return fs.contiguous(),None,cs.contiguous(),normals.contiguous(),None

def comm(a,b,i,j):
  return (a[:,i-1]*b[:,j-1]) - (a[:,j-1]*b[:,i-1])


indexes = [[1,2],[1,3],[1,4],[1,5],[1,6],[2,3],[2,4],[2,5],[2,6],[3,4],[3,5],[3,6],[4,5],[4,6],[5,6]]

indexes_1 = [0,0,0,0,0,1,1,1,1,2,2,2,3,3,4]
indexes_2 = [1,2,3,4,5,2,3,4,5,3,4,5,4,5,5]


def coordinates_1(a,b):
  v = torch.zeros((a.shape[0],15)).cuda()
  for i in range(15):
    v[:,i] = comm(a,b,*indexes[i])

  return v

def coordinates(a,b):

  return (a[:,indexes_1]*b[:,indexes_2] - a[:,indexes_2]*b[:,indexes_1]).reshape((-1,15)).cuda()



def Embed(S1):
  #fs,pts,cs,normals,sum_edges = S1

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



def N_compressed(pt,pt1,w1,w2,sigma):

    gamma = 1 / (2*(sigma * sigma))
    #x, y, b = Vi(0, 3), Vj(1, 3), Vj(2, 3)
    pt = LazyTensor(pt[:,None,:].contiguous())
    pt1 = LazyTensor(pt1[None,:,:].contiguous())

    D = pt.sqdist(pt1)
    K = (-D * gamma).exp()

    w1 = LazyTensor(w1[:,None,:].contiguous())
    w2 = LazyTensor(w2[None,:,:].contiguous())

    ss = ((w1*w2) ).sum()

    res = K*ss
   
    return res.sum(0).sum()


def NC_met_comp(pt,pt1,w1,w2,sigma):

  res = N_compressed(pt,pt1,w1.cuda(),w2.cuda(),sigma)

  return  res

def NC_d_comp(pt,pt1,w1,w2,sigma):
    
    return NC_met_comp(pt,pt,w1,w1,sigma) - (2*NC_met_comp(pt,pt1,w1,w2,sigma)) + NC_met_comp(pt1,pt1,w2,w2,sigma)



def Compute_weights_centres(S1,bdry):

  fs,pts,cs,normals,sum_edges = S1

  #fill,a=Embed(S1)

  if bdry==1.0:

    pt = torch.vstack([pts,cs])
    w1 = Embed(S1)#torch.vstack([a,fill])

  if bdry==0.0:

    pt = cs
    w1 = Embed(S1)

  return pt,w1


def kern_met_torch_2(X,Y,scale):
  """Function to compute kernel matrix of Current in JAX

  Args:
      X (_type_): _description_
      Y (_type_): _description_
      scale (_type_): _description_

  Returns:
      _type_: _description_
  """
  k=3
  X=X.reshape((1,-1,k))
  Y=Y.reshape((1,-1,k))
  ZZ = torch.sum(X*X,axis=2,keepdim=True) + torch.sum(Y*Y,axis=2,keepdim=True).transpose(2,1 )
  SS = -2*X@Y.transpose(2,1)
  fin = torch.exp(- ( torch.abs(SS + ZZ) )/(2*(scale**2) ) )
  return fin



def kern_met_torch(X,Y,scale):
  """Function to compute kernel matrix of Current in JAX

  Args:
      X (_type_): _description_
      Y (_type_): _description_
      scale (_type_): _description_

  Returns:
      _type_: _description_
  """
  k=3
  #X=X.reshape((1,-1,k))
  #Y=Y.reshape((1,-1,k))
  ZZ = torch.sum(X*X,axis=2,keepdim=True) + torch.sum(Y*Y,axis=2,keepdim=True).transpose(2,1 )
  SS = -2*X@Y.transpose(2,1)
  fin = torch.exp(- ( torch.abs(SS + ZZ) )/(2*(scale**2) ) )
  return fin

def GaussKernel(x,y,b,sigma):

    xx = LazyTensor(x[None,:,:])
    yy = LazyTensor(y[:,None,:])
    bb = LazyTensor(b[:,None,:])

    gamma = 1 / (2*(sigma * sigma))
    D2 = xx.sqdist(yy)
    K = (-D2 * gamma).exp()
    #print(K.shape,bb.shape)
    return (K * bb).sum_reduction(axis=0)

def GK(x,y,b,sigma):

    xx = LazyTensor(x[None,:,:])
    yy = LazyTensor(y[:,None,:])
    bb = LazyTensor(b[:,None,:])

    gamma = 1 / (2*(sigma * sigma))
    D2 = xx.sqdist(yy)
    K = (-D2 * gamma).exp()
    #print(K.shape,bb.shape)
    return (K * bb).sum(0)


def kern_metric(X,Y,sigma_w):

  ZZ = torch.sum(X*X,axis=1,keepdims=True) + torch.sum(Y*Y,axis=1,keepdims=True).transpose(1,0)
  #print(X.shape,Y.shape)
  SS = -2*X@Y.transpose(1,0)
  fin = torch.exp(-(SS + ZZ)/(2.0*(sigma_w**2)))
  return fin

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
  k=3
  X=X.reshape((1,-1,k))
  Y=Y.reshape((1,-1,k))
  ZZ = jnp.sum(X*X,axis=2,keepdims=True) + jnp.sum(Y*Y,axis=2,keepdims=True).transpose([0,2,1])
  SS = -2*X@Y.transpose([0,2,1])
  fin = jnp.exp(- ( jnp.abs(SS + ZZ) )/(2*(scale**2) ) )
  return fin[0]


#def Compress_NC(pt,w1,idx,kernel,GK,sigma):

  #ctrl_pts = pt[idx]
  #K_c = kernel(ctrl_pts,ctrl_pts,sigma)
  #K_c = np.array(K_c,dtype=np.float64)

 # vals =  GK(ctrl_pts.reshape((-1,k)),pt.reshape((-1,k)),w1,sigma).cpu().detach().numpy()

#  beta_1 = np.linalg.solve(K_c + 1e-12*np.eye(ctrl_pts.shape[0]),vals) 

  #return ctrl_pts,beta_1

def Compress_NC(pt,w1,idx,kernel,GK,sigma):

  ctrl_pts = pt[idx]
  K_c = kernel(ctrl_pts,ctrl_pts,sigma)
  #K_c = np.array(K_c,dtype=np.float64)

  vals =  GK(ctrl_pts ,pt ,w1,sigma)#.cpu().detach().numpy()

  beta_1 = torch.linalg.solve(K_c + 1e-12*torch.eye(ctrl_pts.shape[0]).cuda(),vals) 

  return ctrl_pts,beta_1











