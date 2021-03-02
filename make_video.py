import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 

from load_data import load_mnist_sevens_nines
from train_mlp import train_mlp 

#Load data:
X_train,Y_train,X_test,Y_test=load_mnist_sevens_nines()
dim_input=50
ignore_pcs=1

#Transform components:
pca=PCA(n_components=dim_input+ignore_pcs)
X_train_transform=pca.fit_transform(X_train.transpose()).transpose()
X_test_transform=pca.transform(X_test.transpose()).transpose()

#Ignore first principal components and permute:
permutation=np.random.permutation(dim_input)
X_train=X_train_transform[ignore_pcs:][permutation]
X_test=X_test_transform[ignore_pcs:][permutation]

#Set hyperparameters:
WEIGHT_SCALE=0.01
N_EPOCHS=100
LR=1e-1
WEIGHT_DECAY=0.0
DIM_LAYERS=[dim_input]+[20]+[1]
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
                                                                                video_length=60)
