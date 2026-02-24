import jax
import torch
import numpy as np
import jax.numpy as jnp
from pyntcloud import PyntCloud
from .Config import *
from jax import config
config.update("jax_enable_x64", True)
import pywavefront
import stripy
from .NC_mets import pre_comp_nc_vec_con,get_parts,Embed


def normalize(shape):
  """function to normalize input shape

  Args:
      shape (_type_): _description_

  Returns:
      _type_: _description_
  """
  return (shape-shape.mean(0))/shape.std(0)


def loader(shape_pars,mode):
   """function to load a given shape

    Args:
        shape_pars (_type_): _description_
        mode (_type_): _description_

    Returns:
        _type_: _description_
    """
    
   if mode=='Currents':
      return Curr_loader(*shape_pars)
   
   if mode=='Varifolds':
     
     return Var_loader(*shape_pars)
   
   
     
   return NC_loader(*shape_pars)

def Curr_loader(gen_pts,M1):
  """Function that takes (shape,mesh) and returns centres and (un)normals 
    for use in Currents

  Args:
      gen_pts (_type_): _description_
      M1 (_type_): _description_

  Returns:
      _type_: _description_
  """
  zz = gen_pts
  v1,v2,v3= zz[M1[:,0]],zz[M1[:,1]],zz[M1[:,2]]


  g1 = v2-v1
  g2 = v3-v1

  v_normals = .5*torch.cross(g1,g2).type(def_type)
  v_centres = (v1 + v2 + v3)/3.0

  alph = v_normals.type(def_type)
  GEN = v_centres.type(def_type)
  n= GEN.shape[0]

  return GEN,alph



def Var_loader(gen_pts,M1):
  """Function that takes (shape,mesh) and returns
     ([centres,unit normals],areas of triangles) for use in Varifolds 

  Args:
      gen_pts (_type_): _description_
      M1 (_type_): _description_

  Returns:
      _type_: _description_
  """
  zz = gen_pts
  v1,v2,v3= zz[M1[:,0]],zz[M1[:,1]],zz[M1[:,2]]


  g1 = v2-v1
  g2 = v3-v1

  v_normals = .5*torch.cross(g1,g2).type(def_type)
  v_centres = (v1 + v2 + v3).type(def_type)/3.0
  v_areas = torch.norm(v_normals,dim=1)
  v_areas_1 = torch.where(v_areas != 0.0,v_areas,np.inf)


  v_norm_sc = v_normals/v_areas_1.unsqueeze(1)

  GEN = torch.hstack((v_centres.type(def_type),v_norm_sc))
  n= GEN.shape[0]

  return GEN,v_areas


def var_proc(shape):
  """ Function taking (shape,mesh) and returns 
      [centres,(un)normals] for use in Varifolds
  Args:
      shape (_type_): _description_

  Returns:
      _type_: _description_
  """
  gen_pts,M1 = shape
  v1,v2,v3= gen_pts[M1[:,0]],gen_pts[M1[:,1]],gen_pts[M1[:,2]]
  g1 = v2-v1
  g2 = v3-v1
  v_normals = .5*torch.cross(g1,g2).type(def_type)
  v_centres = (v1 + v2 + v3).type(def_type)/3.0

  return [v_centres,v_normals]


def var_extract(pars):

  """Function taking (centres,(un)normals) and returns
     [centres,unit normals,areas] for use in varifolds

  Returns:
      _type_: _description_
  """

  x,nu = pars #x is centres (N,3),nu (N,3) is unnomrmalized normals

  N=x.shape[0]

  v_areas = torch.norm(nu,dim=1)
  v_areas_1 = torch.where(v_areas != 0.0,v_areas,np.inf)

  normalized = nu/v_areas_1.unsqueeze(1)
  
  proc_pars = [x,normalized,v_areas]
  return proc_pars


def NC_loader(gen_pts,M1,components,bdry):
  

 parts = get_parts(gen_pts,M1,bdry,*components)

 embedding = Embed(parts)

 fs,pts,cs,normals,sum_edges = parts 
 
 if bdry:
  return  torch.vstack([pts,cs]).contiguous(),embedding.contiguous()
 
 else:
   return  cs,embedding.contiguous()


def pre_comp_nc_vec_con(gen_pts,M1):

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
  bdry_verts = torch.unique(torch.vstack(bdry).reshape((-1,1)))
  bdry_stack = torch.vstack(bdry)

  bdry_vert_ed_inds = np.vstack([ np.where((bdry_stack == bdry_verts[i]).sum(1)) for i in range(len(bdry_verts ))])
  bdry_vert_edges = bdry_stack[bdry_vert_ed_inds]
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
