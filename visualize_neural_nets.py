import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.colors import Normalize
from utils import give_ranks_of_array


def visualize_neural_net(ax, left, right, bottom, top, layer_list,weight_list,cmap=None,ranks=False,threshold=None):
    '''
    Function to visualize a multilayer perceptron (fully connected neural network)
    This is a modification of https://gist.github.com/craffel/2d727968c3aaebd10359
    Usage by:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> visualize_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2], weight_list)
    
    Input:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_list : list of int
            List of layer sizes, including input and output dimensionality
        - weight_list : list of np.array of shape (layer_size[it],layer_size[it+1])
                        lift of weight matrices 
    Output: 
        modified ax element
    '''
    ax.axis('off')
    #Set spacing:
    n_layers = len(layer_list)
    v_spacing = (top - bottom)/float(max(layer_list))
    h_spacing = (right - left)/float(len(layer_list) - 1)

    # Nodes
    for n, layer_size in enumerate(layer_list):
        
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.

        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_list[:-1], layer_list[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.

        #Set normalization:
        if ranks:
            color_scale=give_ranks_of_array(weight_list[n])
        else:
            color_scale=weight_list[n]
        
        vmin=color_scale.min()
        vmax=color_scale.max()

        if threshold is not None:
            vmin=threshold*vmax+(1-threshold)*vmin
        norm = Normalize(vmin=vmin, vmax=vmax,clip=True)

        for m in range(layer_size_a):
            for o in range(layer_size_b):
                
                weight=color_scale[o,m]

                if ranks or weight>vmin:
                    line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing],c='k')
                
                    line.set_color(color=cmap(norm(weight)))
                    ax.add_artist(line)
    
    return(ax)



'''
#Plot example:
fig = plt.figure(figsize=(12, 12))

#Number of layers:
dim_layers=[30,30,15,10]
n_layers=len(dim_layers)-1
weight_scale=0.1

#Initialize random weight list:
weight_list=[np.random.normal(size=(dim_layers[it+1],dim_layers[it]),loc=0.,
                        scale=weight_scale) for it in range(n_layers)]

visualize_neural_net(fig.gca(), .01, .99, .01, .99, dim_layers, weight_list,cmap=plt.cm.coolwarm,threshold=0.2)

plt.tight_layout()
plt.savefig("plots/illustrate_visualization.pdf")
'''