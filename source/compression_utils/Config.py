#%%
import jax
import torch
import numpy as np
import jax.numpy as jnp
from pyntcloud import PyntCloud
from jax import config
config.update("jax_enable_x64", True)
import pywavefront
import stripy
from .NC_mets import pre_comp_nc_vec_con

#%%
torch.set_default_dtype(torch.float32)
def_type = torch.get_default_dtype()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
def normalize(shape):
    return (shape-shape.mean(0))/shape.std(0)



k = 3

if k==2:

  gen_pts = np.loadtxt('plane.txt').T
  gen_pts=torch.tensor(gen_pts).type(def_type).cuda()
  n = gen_pts.shape[0]


if k==3:
  
  gen_name = 'sphere'


  if gen_name=='sphere':
     
     S = stripy.spherical_meshes.random_mesh(number_of_points=5000)
     gen_pts = torch.tensor(S.points).type(def_type)#conv.points
     M1 = np.array(S.simplices)

  else:
    pc1 = pywavefront.Wavefront(f'../data/{gen_name}.obj',create_materials=True,collect_faces=True)
    gen_pts = torch.tensor(pc1.vertices).type(def_type)
    M1 =np.array(pc1.mesh_list[0].faces)

  targ_name='/mnt/c/Users/allen/Desktop/PhD/Numerical exp/Compression/data/Curr_var_shapes/BrainLow.stl'
  
  
  if 'stl' in targ_name:
    from stl import mesh
    import trimesh

    myobj = trimesh.load_mesh(f'{targ_name}', enable_post_processing=True, solid=True) # Import Objects

    target = torch.tensor(myobj.vertices).reshape((-1,3)).type(def_type)
    M2 = np.array(myobj.faces)

  if 'obj' in targ_name:
    
    pc2 = pywavefront.Wavefront(f'{targ_name}',create_materials=True,collect_faces=True)
    target = torch.tensor(pc2.vertices).type(def_type)
    M2 =np.array(pc2.mesh_list[0].faces)

  else:

    pc2 = pywavefront.Wavefront(f'../data/{targ_name}',create_materials=True,collect_faces=True)
    target = torch.tensor(pc2.vertices).type(def_type)
    M2 =np.array(pc2.mesh_list[0].faces)
    


gen_pts=1.*normalize(gen_pts)
target=1.*normalize(target)

shape_bdry = False
target_bdry = False

shape_pars = [gen_pts.cuda(),np.array(M1)]#,
              # pre_comp_nc_vec_con(gen_pts,torch.tensor(M1),shape_bdry ),shape_bdry ]
target_pars = [target.cuda(),np.array(M2)]#,
              #  pre_comp_nc_vec_con(target,torch.tensor(M2),target_bdry ),target_bdry ]
#%%



Ps =  [1000]

trials=1
sqs = []
betas = []


sampler = 'DAC' #choose RLS or DAC methods
mode='Varifolds' #choose NC,Varifolds or Currents


kernel_name = 'gaussian' # spatial jernel

#hyperparams for kenrels
sigma_w = .4
sigma_sph = .5

kernel_param = [sigma_w,sigma_sph]#[sigma_w]# 
scales =  torch.tensor([1.,.5,.1,.05],dtype=def_type,device=device)

kern_params= kernel_param


exp_pars=Ps,trials,sampler,mode,kern_params

















