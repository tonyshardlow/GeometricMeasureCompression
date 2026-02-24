#%%
import time

import jax.numpy as jnp
import numpy as np
import torch
from jax import config

from source.compression_utils import Var_Metrics
from source.compression_utils.Compressors import orthog_pursuit, var_quantizer
from source.compression_utils.Config import (M1, M2, Ps, device, gen_pts, k,
                                             kern_params, kernel_param, mode,
                                             sampler, scales, shape_pars,
                                             target_pars, trials)
from source.compression_utils.DAC import DAC
from source.compression_utils.Experiments import Experiment
from source.compression_utils.Geodesic_old import Geodesic_shooting
from source.compression_utils.preproc import var_extract, var_proc

config.update("jax_enable_x64", True)


# %%
times = []
for P in Ps:

    exp_pars = [P], trials, sampler, mode, kern_params

    start = time.time()
    res = Experiment(exp_pars, shape_pars)
    end = time.time()
    diff = end-start
    times.append(diff)
    print('compression time taken: ', diff, "s")
# %%


betas, ctrl_pts, idx = res[0], res[1][0][0], res[1][0][1]

err = Var_Metrics.Err_var([ctrl_pts[:, :3].type(torch.float32).cuda(), ctrl_pts[:, 3:].type(torch.float32).cuda(),
                          torch.tensor(np.array(betas[0]), dtype=torch.float32
                                       ).cuda()], target_pars, kern_params)

print("Compression error is: ", err)
# %%


target_comp = var_proc(target_pars)

a = np.round(np.linspace(
    0, target_comp[0].shape[0] - 1, target_comp[0].shape[0])).astype(int)[:]

inds = np.random.choice(a, size=Ps[0], replace=False)


p1 = target_comp[0][inds].cuda()
p2 = target_comp[1][inds].cuda()


par_opt = torch.stack([p1, p2])

quant_res = var_quantizer(par_opt, target_comp, kernel_param, 1, 1e-2)
# %%
ctrl_pts, betas = quant_res[0], quant_res[1]

quant_err = Var_Metrics.Err_var([ctrl_pts[:, :3].type(torch.float32).cuda(), ctrl_pts[:, 3:].type(torch.float32).cuda(),
                                 betas], target_pars, kern_params)


# %% run experiment here


start = time.time()
res = Experiment(exp_pars, shape_pars)
end = time.time()

print(end-start)

# %%
orthog_pursuit(kern_params[:1], shape_pars, 2)
# %%
target = var_proc([gen_pts, np.array(M1)])

inds = np.round(np.linspace(0, target[0].shape[0] - 1, Ps[0])).astype(int)[:]

p1 = target[0][inds].cuda()
p2 = target[1][inds].cuda()


par_opt = torch.stack([p1, p2])

start = time.time()
res = var_quantizer(par_opt, target, kern_params, 50)
end = time.time()

print(end-start)
# %%
# call experiment to compress both target and gen_ps first (weights of gen_pts dont matter here)
# compress genpts
_, _, idx = Experiment(exp_pars, shape_pars)
# compress target
a, b, _ = var_extract(var_proc([target, M2]))
betas, _, idx1 = Experiment(exp_pars, target_pars)


targ_pars = [a[idx1].cuda(), b[idx1].cuda().type(torch.float32), torch.tensor(
    np.array(betas[0][1])).type(torch.float32).cuda()]


p_size = 5000

scores = DAC(gen_pts.cpu().detach().numpy(), .1,
             1000, Var_Metrics.kern_m, kernel_param)
ctrl_idx = np.random.choice(gen_pts.shape[0], p_size, replace=False, p=jnp.array(
    scores/scores.sum(), dtype=jnp.float32))

p_init = torch.zeros((p_size, k), device=device).reshape((-1, k))
p_init.requires_grad = True

Shooter = Geodesic_shooting([gen_pts.cuda(), M1, idx, ctrl_idx],
                            targ_pars, scales, p_init, [mode, kernel_param])
# %%
Shooter.optimize(1)
