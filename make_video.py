import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 

from data.load_data import load_mnist_sevens_nines
from train_mlp import train_mlp 

#Load data:
X_train,Y_train,X_test,Y_test=load_mnist_sevens_nines()
dim_input=40
permutation=np.random.permutation(dim_input)
pca=PCA(n_components=dim_input)
X_train=pca.fit_transform(X_train.transpose()).transpose()[permutation]
X_test=pca.transform(X_test.transpose()).transpose()[permutation]

#Set hyperparameters:
WEIGHT_SCALE=0.01
N_EPOCHS=500
LR=1e-2
WEIGHT_DECAY=0.
DIM_LAYERS=[dim_input]+[40,40]+[1]
VIS_SUBSET=DIM_LAYERS

weight_list,train_loss_list,train_acc_list,test_loss_list,test_acc_list=train_mlp(weight_scale=WEIGHT_SCALE,
                                                                                dim_layers=DIM_LAYERS,
                                                                                n_epochs=N_EPOCHS,
                                                                                weight_decay=WEIGHT_DECAY,
                                                                                lr=LR,
                                                                                X_train=X_train,
                                                                                Y_train=Y_train,
                                                                                X_test=X_test,
                                                                                Y_test=Y_test,
                                                                                print_every=10,
                                                                                make_video=True,
                                                                                vis_subset=VIS_SUBSET,
                                                                                video_name='test',
                                                                                plot_every=1,
                                                                                video_length=3)
