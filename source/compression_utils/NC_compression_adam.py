
#%% LOADING AND PREPROCESSING
import pykeops
import jax
import torch
from pykeops.torch import LazyTensor,Genred,Vi,Vj,Pm
from pykeops.numpy import LazyTensor as LazyTensor_np
import numpy as np
import jax.numpy as jnp
from NC_compression_utils_adam import *
import pywavefront

torch.set_default_dtype(torch.float32)
def_type = torch.get_default_dtype()

#shape data format
format='obj'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load in data
pc1 =   pywavefront.Wavefront(f'1.obj',create_materials=True,collect_faces=True) #
S1 =  torch.tensor(pc1.vertices)
MM = np.array(pc1.mesh_list[0].faces)#


pc2 = pywavefront.Wavefront(f'2.obj',create_materials=True,collect_faces=True)   #
S2 = torch.tensor(pc2.vertices)
MM1 = np.array(pc2.mesh_list[0].faces)#

template = normalize(S1).type(def_type).cuda()
target = normalize(S2).type(def_type).cuda()

## if the shapes to be compared have a topological boundary
bdry = False

"""" load in before computation, only needs to be done once as preprocessing. Note: deformations of template also have same
temp_struct so do not need to recompute for deformations"""
gen_struct = pre_comp_nc_vec_con(template,torch.tensor(MM),bdry)
targ_struct= pre_comp_nc_vec_con(target,torch.tensor(MM1),bdry)

#%% COMPRESSION AND COMPUTATION

#NC space lengthscale
sig=torch.tensor([.4]).cuda()


#shape info required for NC metric
S1 = [gen_struct ,template.cuda(),MM]
S2 = [targ_struct ,target.cuda(),MM1]


gen_struct,gen_pts,MM = S1[0],S1[1],S1[2]
target_struct,target,MM1 = S2[0],S2[1],S2[2]

#use preprocessing info to get shape specific quantities
gen_parts = get_parts(gen_pts,torch.tensor(MM),bdry,*gen_struct )
targ_parts = get_parts(target,torch.tensor(MM1),bdry,*target_struct )

#compute NC weights for each shape
targ_weights = Embed(targ_parts)
gen_weights = Embed(gen_parts)

#get NC dirac delta centres for each shape
_,pts,cs,_,_  = targ_parts
_,pts1,cs1,_,_  = gen_parts


if bdry:
    targ_centres = torch.vstack([pts,cs])
    gen_centres = torch.vstack([pts1,cs1])

else:
    targ_centres = cs
    gen_centres = cs1

#if compressing and sampler
compress = True
sampler = 'DAC'#'uni'

#print(targ_centres.shape)
#desired compression size for both shapes if compressing (max is  the size of gen_centres or targ_centres ), can choose different sizes as well for each
size=500



if compress:

    #uniformly sample delta centres
    if sampler=='uni':
        idx_gen =  np.random.choice(gen_centres.shape[0],size=size,replace=False)
        idx_targ = np.random.choice(targ_centres.shape[0],size=size,replace=False)

    #use DAC RLS Sampler instead
    elif sampler=='DAC':
        #.1 ,1000 are hyperparams of the DAC chosen heuristically it is NOT the comp size

        #compute RLS scores
        scores_gen=DAC_vec(gen_centres, .1, 1000, kern_met_NC, [sig])
        #sample
        idx_gen=np.random.choice(gen_centres.shape[0],size,replace=False,p=jnp.array(scores_gen/scores_gen.sum()))

        #compute RLS scores
        scores_targ = DAC_vec(targ_centres, .1, 1000, kern_met_NC, [sig])
        #sample
        idx_targ=np.random.choice(targ_centres.shape[0],size,replace=False,p=jnp.array(scores_targ/scores_targ.sum()))

    #get compressed centres and weights of targ shape
    targ_comp_centres = targ_centres[idx_targ]
    targ_comp_weights = weight_comp(idx_targ,targ_centres,targ_weights,sig)

    #get compressed centres and weights of gen shape
    gen_comp_centres = gen_centres[idx_gen]
    gen_comp_weights = weight_comp(idx_gen,gen_centres.contiguous(),gen_weights.contiguous(),sig.contiguous())

    #compute NC metric post compression
    res = NC_metric(gen_comp_centres.contiguous(),gen_comp_weights.contiguous(),targ_comp_centres.contiguous(),targ_comp_weights.contiguous(),sig.contiguous()).contiguous()

else:
    #if not compress compute full NC_metric
    res = NC_metric(gen_centres.contiguous(),gen_weights,targ_centres.contiguous(),targ_weights.contiguous(),sig.contiguous()).contiguous()


print("NC metric is ",res)


