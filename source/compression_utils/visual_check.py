#%%

import torch 
from Visuals import *
import trimesh
import numpy as np
import stripy
#%%
load_path = '/mnt/c/Users/allen/Desktop/PhD/Numerical exp/Compression/NC-shapes/Queen.stl'

torch_path = '/mnt/c/Users/allen/Desktop/PhD/Numerical exp/Compression/NC-shapes/pumpkin-full-out'

myobj = trimesh.load_mesh(f'{load_path}', enable_post_processing=True, solid=True) # Import Objects

target = torch.tensor(myobj.vertices).reshape((-1,3)).type(torch.float32)
M2 = np.array(myobj.faces)

shape_load = torch.load(torch_path)

shapes = [shape_load]
meshes = []

S = stripy.spherical_meshes.random_mesh(number_of_points=5000)
M1 = np.array(S.simplices)
#%%
figure = visualise(shapes,meshes)

figure.update_layout(
    scene = dict(
        xaxis = dict(visible=False),
        yaxis = dict(visible=False),
        zaxis =dict(visible=False)
        )
    )

figure.write_html("pumpkin_full.html")