
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

from forward_backward import forward, backward, loss_func, accuracy 
from visualize_neural_nets import visualize_neural_net

def train_mlp(weight_scale,dim_layers,n_epochs,weight_decay,lr,
                X_train,Y_train,X_test,Y_test,print_every=100,
                make_video=False,vis_subset=None,video_name=None,plot_every=None,video_length=None):

    n_layers=len(dim_layers)-1
    
    #Initialize random weight list:
    weight_list=[np.random.normal(size=(dim_layers[it+1],dim_layers[it]),loc=0.,
                            scale=weight_scale) for it in range(n_layers)]

    train_loss_list=[]
    train_acc_list=[]
    val_loss_list=[]
    val_acc_list=[]


    if make_video:
        xlim=[0,n_epochs+3]
        ylim=[0.4,1.]

        #Setup video:
        folder = 'videos/'
        filename=folder+'random.png'

        #Create figure of inital network:
        fig = plt.figure(figsize=(12, 12))
        #fig,ax = plt.subplots(ncols=2,nrows=1,figsize=(24, 12))
        #ax[]
        weight_list_visualize=[weight_list[j][:,:vis_subset[j]] for j in range(len(weight_list)-1)]+[weight_list[-1]]
        visualize_neural_net(fig.gca(), .1, .9, .1, .9, vis_subset,weight_list_visualize)
        #visualize_neural_net(fig.gca(), .1, .9, .1, .9, vis_subset,weight_list_visualize)
        #ax[1].set_xlim(xlim)
        #ax[1].set_ylim(ylim)

        plt.savefig(filename)

        frame = cv2.imread(filename)
        height, width, layers = frame.shape
        
        fps=max(int((n_epochs+1)/video_length),1)
        print("Frames per second: ", fps)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(folder+video_name+'.avi',fourcc,fps,(width, height))
        video.write(cv2.imread(filename))
        #cv2.waitKey(wait_per_frame)
        plt.close()


    for it in range(n_epochs):

        #Backpropagation: 
        weight_grad_list,loss,acc=backward(weight_list,X_train,Y_train)   
        
        #Compute output on test set:
        output,activ_list=forward(weight_list,X_test)

        #Get loss and accuracy on test set:
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
                fig = plt.figure(figsize=(12, 12))
                #fig = plt.figure(figsize=(12, 12))

                weight_list_visualize=[weight_list[j][:,:vis_subset[j]] for j in range(len(weight_list)-1)]+[weight_list[-1]]            
                visualize_neural_net(fig.gca(), .1, .9, .1, .9, vis_subset,weight_list_visualize)#,norm=norm)
                fig.suptitle("Accuracy: %.4f"%acc_test)
                #ax[1].plot(list(range(it)),val_acc_list)    
                plt.savefig(filename)
                video.write(cv2.imread(filename))
                #cv2.waitKey(wait_per_frame)
                plt.close()

    if make_video:
        cv2.destroyAllWindows()
        video.release()

    return(weight_list,train_loss_list,train_acc_list,val_loss_list,val_acc_list)