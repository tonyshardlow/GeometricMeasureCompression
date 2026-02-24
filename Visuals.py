import plotly
import plotly.graph_objs as go

##done

asp,pos,light = dict(x=5, y=5, z=5),dict(x=100,y=100, z=10),dict(ambient=.5,
    diffuse=1,
    fresnel=4,
    specular=0.5,
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
            k=meshes[i][:, 2],opacity=.5,lighting=lights,lightposition=position
            ) for i in range(len(shapes))]

    fig = go.Figure(data = data)

    fig.update_layout(
        scene = dict(
            aspectratio=aspect
        )
    )

    return fig



