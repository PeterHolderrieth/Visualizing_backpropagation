
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

from forward_backward import forward, backward, loss_func, accuracy 
from visualize_neural_nets import visualize_neural_net

'''
This script provides a function which trains a multilayer perceptron (fully connected
neural network). Additionally, it provides a function which creates a video visualizing
the training by plotting the neural network over time.
'''


def train_mlp(weight_scale,dim_layers,n_epochs,weight_decay,lr,
                X_train,Y_train,X_test,Y_test,print_every=100,
                make_video=False,vis_subset=None,video_name=None,plot_every=None,video_length=None,prop_weight_to_grad=3):

    #Number of layers:
    n_layers=len(dim_layers)-1
    
    #Initialize random weight list:
    weight_list=[np.random.normal(size=(dim_layers[it+1],dim_layers[it]),loc=0.,
                            scale=weight_scale) for it in range(n_layers)]

    #Initialize logging lists:
    train_loss_list=[]
    train_acc_list=[]
    val_loss_list=[]
    val_acc_list=[]


    if make_video:
        #Setup video
        folder = 'videos/'
        filename=folder+'random.png'

        figsize=(12, 6)
        #Create figure of inital network
        fig,ax = plt.subplots(ncols=2,nrows=1,figsize=figsize)
        
        #Create subset of weights to visualize
        weight_list_visualize=[weight_list[j][:,:vis_subset[j]] for j in range(len(weight_list)-1)]+[weight_list[-1]]
        visualize_neural_net(ax[0], .01, .99, .01, .99, vis_subset,weight_list_visualize,cmap=plt.cm.copper,ranks=False,threshold=0.7)
        
        #Axis limits for logging
        xlim=[0,n_epochs]
        ylim=[0.5,1]

        #Create initial logging plot
        it_vec=np.arange(len(val_acc_list))
        ax[1].plot(it_vec,train_acc_list,color='blue',label='train')  
        ax[1].plot(it_vec,val_acc_list,color='orange',label='test')  
        ax[1].set_xlim(xlim)
        ax[1].set_ylim(ylim)
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("accuracy")
        ax[1].legend()

        #Create initial figure and get geometry
        plt.savefig(filename)
        frame = cv2.imread(filename)
        height, width, layers = frame.shape
        
        #Number of frames per iteration
        n_frames_per_it=1+prop_weight_to_grad
        #Number of total frames
        n_frames_total=n_epochs*n_frames_per_it+1
        #Frames per second:
        fps=max(int(n_frames_total/video_length),1)

        #Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(folder+video_name+'.avi',fourcc,fps,(width, height))

        #Start with weight frame (twice the time to begin with)
        for it in range(2*prop_weight_to_grad):
            video.write(cv2.imread(filename))
        
        #Close figure
        plt.close()


    for it in range(n_epochs):

        #Backpropagation
        weight_grad_list,loss,acc=backward(weight_list,X_train,Y_train)   
        
        #Compute output on test set
        output,activ_list=forward(weight_list,X_test)

        #Get loss and accuracy on test set
        loss_test=loss_func(output,Y_test)
        acc_test=accuracy(output,Y_test)
        
        if it%print_every==0:
            print("Epoch %3d || Loss: %.4f || Accuracy: %.4f || Test loss: %.4f || Test accuracy: %.4f"%(it,loss,acc,loss_test,acc_test))
        
        #Peform gradient descent step:
        for it in range(n_layers):
            weight_list[it]=(1-weight_decay)*weight_list[it]-lr*weight_grad_list[it]
        
        train_loss_list.append(loss)
        train_acc_list.append(acc)
        val_loss_list.append(loss_test)
        val_acc_list.append(acc_test)

        if make_video:
            if it%plot_every==0:
                
                #--------------------------
                #1. GRADIENT PLOT
                #--------------------------
                fig,ax = plt.subplots(nrows=1,ncols=2,figsize=figsize)

                #Visualize gradients:
                weight_list_visualize=[weight_grad_list[j][:,:vis_subset[j]] for j in range(len(weight_list)-1)]+[weight_list[-1]]            
                visualize_neural_net(ax[0], .01, .99, .01, .99, vis_subset,weight_list_visualize,cmap=plt.cm.turbo,ranks=True)
                
                #Visualize logging:
                it_vec=np.arange(len(val_acc_list))
                ax[1].plot(it_vec,train_acc_list,color='blue',label='train')  
                ax[1].plot(it_vec,val_acc_list,color='orange',label='test')  
                ax[1].set_xlim(xlim)
                ax[1].set_ylim(ylim)
                ax[1].set_xlabel("epoch")
                ax[1].set_ylabel("accuracy")
                ax[1].legend()
                
                #Create suptitle:
                fig.suptitle("%10s || Accuracy: %.4f"%("Gradients",acc_test))

                #Save to video writer:
                plt.savefig(filename)
                video.write(cv2.imread(filename)) 
                plt.close()
                
                #--------------------------
                #2. WEIGHT PLOT
                #--------------------------
                fig,ax = plt.subplots(nrows=1,ncols=2,figsize=figsize)
                
                #Visualize network:
                weight_list_visualize=[weight_list[j][:,:vis_subset[j]] for j in range(len(weight_list)-1)]+[weight_grad_list[-1]]            
                visualize_neural_net(ax[0], .01, .99, .01, .99, vis_subset,weight_list_visualize,cmap=plt.cm.copper,ranks=False,threshold=0.7)
                fig.suptitle("%10s || Accuracy: %.4f"%("Weights",acc_test))
                
                ax[1].plot(it_vec,train_acc_list,color='blue',label='train')  
                ax[1].plot(it_vec,val_acc_list,color='orange',label='test')  
                ax[1].set_xlim(xlim)
                ax[1].set_ylim(ylim)
                ax[1].set_xlabel("epoch")
                ax[1].set_ylabel("accuracy")
                ax[1].legend()


                plt.savefig(filename)
                for it in range(prop_weight_to_grad):
                    video.write(cv2.imread(filename))
                plt.close()
                

    if make_video:
        cv2.destroyAllWindows()
        video.release()

    return(weight_list,train_loss_list,train_acc_list,val_loss_list,val_acc_list)
