import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.colors import Normalize

def visualize_neural_net(ax, left, right, bottom, top, layer_list,weight_list=None,norm=None,cmap=None):
    '''
    Function to visualize a multilayer perceptron (fully connected neural network)
    This is a modification of https://gist.github.com/craffel/2d727968c3aaebd10359
    Usage by:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
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
    #Create scaling for weights
    if cmap is None:
        cmap = plt.cm.hot
    
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
        
        if weight_list is not None:
            vmax=weight_list[n].max()#0.5*(weight_list[n].max()-weight_list[n].min())+weight_list[n].min()
            vmin=weight_list[n].min()
            norm = Normalize(vmin=vmin, vmax=vmax,clip=True)

        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing],c='k')
                
                if weight_list is not None:                    
                    weight=weight_list[n][o,m]
                    line.set_color(color=cmap(norm(weight)))
                    line.set_alpha(0.5)
                ax.add_artist(line)
    
    return(ax)

# def generate_imagefiles(img,folder):
#     for i in range(len(img)):
#         plt.imshow(img[i], cmap=plt.cm.Greys_r)
#         plt.savefig(folder + "/file%02d.png" % i)

    #os.chdir(folder)
    # subprocess.call([
    #     'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
    #     'video_name.mp4'
    # ])
    # for file_name in glob.glob("*.png"):
    #     os.remove(file_name)

