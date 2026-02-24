#%%
from source.compression_utils.Var_Metrics import E_var_keops_normals,kern_m
import torch
import numpy as np
#%%
#check shapes

def test_shape_kernel():
    n,m=1000,500
    XX,YY= np.random.randn(n,3),np.random.rand(m,3)
    sigma,sig=.4,.5

    result = kern_m(XX, YY, sigma, sig)

    assert result.shape==(n,m)


#check positive current metric

def test_var_metrics_positive():
    v_centres, v_norm_sc, v_weights = torch.randn(1000,3),torch.abs(torch.randn(1000,1)),torch.randn(1000,3)
    w_centres, w_norm_sc, w_weights = torch.randn(1000,3),torch.abs(torch.randn(1000,1)),torch.randn(1000,3)
    
    scales = 0.4,0.5
    
    v_pars = v_centres, v_norm_sc, v_weights 
    w_pars = w_centres, w_norm_sc, w_weights

    result = E_var_keops_normals(v_pars, w_pars, scales)
    
    assert result > 0


