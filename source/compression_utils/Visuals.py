import plotly
import plotly.graph_objs as go
import pywavefront
from pyntcloud import PyntCloud
import torch
import numpy as np
##done

asp,pos,light = dict(x=5, y=5, z=5),dict(x=0,
              y=-10,
              z=0),dict(ambient=0.2,
            diffuse=1,
            fresnel=4,
            specular=1.,
            roughness=0.05)


def visualise(shapes,meshes,options=[asp,pos,light]):
    
    """ Function to visualise input 3d shapes via plotly

    Args:
        shapes (list): list of point clouds
        meshes (list): list of triangulation indices
        options (list): list of dicts containing plot options
                        for aspect ratio, position and lighting resp

    Returns:
        plotly.fig: plotly 3DMESHplot of input shapes via plotly
    """
    aspect,position,lights = options

    shapes = [shapes[i] for i in range(len(shapes)) ]
    meshes = [meshes[i] for i in range(len(meshes))]


    data=[go.Mesh3d(x=shapes[i][:,0], y=shapes[i][:,1], z=shapes[i][:,2], i=meshes[i][:, 0],
            j=meshes[i][:, 1],
            k=meshes[i][:, 2]
            ) for i in range(len(shapes))]
    #data=[go.Mesh3d(x=shapes[i][:,0], y=shapes[i][:,1], z=shapes[i][:,2],lighting=lights,lightposition=position
    #        ) for i in range(len(shapes))]
    #print(len(data))
    fig = go.Figure(data = data)

    # fig.update_layout(
    #     scene = dict(
    #         aspectratio=aspect
    #     )
    # )

    return fig



if __name__ == '__main__':

    print(torch.load('/mnt/c/Users/allen/Desktop/PhD/Numerical exp/Compression/data/-20_human_missing.pt'))

    paths=['/mnt/c/Users/allen/Desktop/PhD/Numerical exp/Compression/data/_ (103).obj',
        '/mnt/c/Users/allen/Desktop/PhD/Numerical exp/Compression/data/_ (103).ply', 
        '/mnt/c/Users/allen/Desktop/PhD/Numerical exp/Compression/data/_ (113).ply'
        ]


    for i,path in enumerate(paths):
        if '.ply' in path:
            

            im = PyntCloud.from_file(path) 
    
            target=np.array(im.points)
            M2 = np.array(im.mesh )

        


        else:
                        
            pc2 = pywavefront.Wavefront(path,create_materials=True,collect_faces=True)
            target = np.array(torch.tensor(pc2.vertices))#.type(def_type)
            M2 =np.array(pc2.mesh_list[0].faces)
            #print(target.shape,M2.shape)


        figure=visualise([target],[M2] )
        
        save_path=f'/mnt/c/Users/allen/Desktop/PhD/Numerical exp/Compression/data/{i}.html'

        figure.write_html(save_path,auto_open=False)