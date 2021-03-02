import numpy as np 
import matplotlib.pyplot as plt


EPS=1e-8

def relu(x):
    '''
    ReLU activation function.
    x - np.array
    '''
    return(np.maximum(x,0))

def relu_deriv(relu_x):
    '''
    Backward (derivative) of ReLU activation function.
    relu_x - np.array - relu of some array x
    '''
    return((relu_x>0).astype(float))

def relu_back(arr_in,grad_out):

    return(relu_deriv(arr_in)*grad_out)

def sigmoid(x,eps=EPS):
    '''
    Sigmoid activation function.
    x - np.array 
    '''
    return 1/(1+np.exp(-x+eps))

def sigmoid_deriv(sigmoid_x):
    '''
    Backward (derivative) of sigmoid activation function.
    sigmoid_x - np.array - sigmoid of some array x
    '''
    return(sigmoid_x*(1-sigmoid_x))

def sigmoid_back(arr_in,grad_out):
    
    return(sigmoid_deriv(arr_in)*grad_out)

def loss_func(x,y,eps=EPS):
    '''
    Loss function. Here: cross entropy error.
    x,y - np.array's of same shape, x gives the probability of label 1, y gives true labels +-1
    eps - float>0 - to ensure numerical stability at taking np.log()
    '''
    #Express labels in zeros and ones:
    cross_entr=-np.log(x+eps)*y-np.log(1-x+eps)*(1-y)
    return(np.mean(cross_entr))

def loss_back(x,y,eps=EPS):
    '''
    Backward (derivative) of loss function. Here: mean squared error (scaled by factor 0.5)
    x,y - np.array's of same shape
    eps - float>0 - to ensure numerical stability at division by zero
    '''
    #Express labels in zeros and ones:
    return ((-y/(x+eps)+(1-y)/(1-x+eps))/x.size)

def accuracy(x,y):
    true_positives=np.sum((x>0.5).astype(float)*y)
    true_negatives=np.sum((x<0.5).astype(float)*(1-y))
    acc=(true_negatives+true_positives)/x.size
    return(acc)

def forward(weight_list,data):
    '''
    Input:
        weight_list - list of np.arrays of shape (n[i+1],n[i]) for (n[1],...,n[k]) with n[1]=dim_input
        data - np.array of shape (dim_input,n_samples)
    Output: 
        output - shape (n[k],n_samples)
        activ_list - list of all activations of layers across samples with shape (n[i],n_samples)
    '''
    #Init
    activ_list=[data]
    n_layers=len(weight_list)
    
    #Pass through hidden layers
    for it in range(n_layers-1):
        hidden_pre_activation=np.dot(weight_list[it],activ_list[-1])
        hidden_activation=relu(hidden_pre_activation)
        activ_list.append(hidden_activation)
    
    #Pass through final layer
    output=sigmoid(np.dot(weight_list[-1],activ_list[-1]))
    activ_list.append(output)
    
    return(output,activ_list)


def backward(weight_list,data,labels):
    '''
    Input:
        weight_list - list of np.arrays of shape (n[i+1],n[i]) for (n[1],...,n[k]) with n[1]=dim_input
        data - np.array of shape (dim_input,n_samples)
        labels - np.array of shape (n_samples) - true labels
    Output: 
        weight_grad_list - list of arrays - same shape as weight_list, gradients of weights
        loss - float - loss on data with respect to labels
    '''
    
    #Compute forward:
    output,activ_list=forward(weight_list,data)

    n_layers=len(weight_list)

    #Create two empty lists:
    activ_grad_list=[None for it in range(n_layers+1)]
    weight_grad_list=[None for it in range(n_layers)]

    #Get loss and initial gradient:
    loss=loss_func(output,labels)
    acc=accuracy(output,labels)

    loss_grad=loss_back(output,labels)
    
    activ_grad_list[-1]=sigmoid_back(output,loss_grad)
   
    for it in range(n_layers):
        
        #Get activation of previous layer and the gradient of the subsequent layer:
        prev_activ=activ_list[-it-2]
        out_grad=activ_grad_list[-it-1]

        #Compute the gradient with respect to the weights:
        #(Outer product + average over the different training samples):
        weight_grad=np.dot(out_grad,prev_activ.transpose())
        weight_grad_list[-it-1]=weight_grad

        #Compute the gradient with respect to the previous activation:
        grad_post_activ=np.dot(weight_list[-it-1].transpose(),activ_grad_list[-it-1])
        grad_hidden=relu_back(activ_list[-it-2],grad_post_activ)

        activ_grad_list[-it-2]=grad_hidden
        
    return(weight_grad_list,loss,acc)

