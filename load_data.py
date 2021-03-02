import scipy.io 

#Hyperparameters:
FILEPATH='data/mnist_sevens_nines.mat'

def load_mnist_sevens_nines():
    #Load train and test files:
    mat = scipy.io.loadmat(FILEPATH)

    X_train=mat['X_train']
    Y_train=mat['y_train']

    #Reshape to 0/1-labels (from -1/1 labels)
    Y_train=0.5*(Y_train+1)

    X_test=mat['X_test']
    Y_test=mat['y_test']

    #Reshape to 0/1-labels (from -1/1 labels)
    Y_test=0.5*(Y_test+1)

    return(X_train,Y_train,X_test,Y_test)