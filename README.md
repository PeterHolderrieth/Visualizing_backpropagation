<img src="https://github.com/PeterHolderrieth/backpropagation/blob/master/videos/final_file.gif">

# Visualizing backpropagation

This repostiory serves as an illustration for the classical backpropagation algorithm, 
one of the most important machine learning concepts. It is meant to be useful for students studying the subject. 
We provide:

1. A **visualization** tool for neural networks 
2. A visualization tool for the **training** of neural networks giving illustrative videos.
3. An implementation of the backpropagation algorithm for fully connected neural networks.


## 1. Visualizing neural networks

The visualization tool for fully connected neural networks allows to plot a fully connected neural network with connections colored accordings to weights. 
The code can be found in the ```visualize_neural_nets.py``` file.

<img src="https://github.com/PeterHolderrieth/backpropagation/blob/master/plots/illustrate_visualization.png" width="300" height="300">

## 2. Visualizing training of neural networks

We also provide a tool to visualize training (see gif above) which illustrates gradient updates 
and changes of weights. For a full video, see https://user-images.githubusercontent.com/57487578/109711025-4c478b00-7b96-11eb-8f77-c13cc3f1ede1.mp4.
The code is integrated in the ```train_mlp.py``` file.

## 3. Backpropagation algorithm

The backpropagation algorithm can be found in the ```train_mlp.py``` file. We did not use automatic differentiation libraries such as TensorFlow
since this repository is meant for teaching and illustration only.

## Data 

As a simple example data set, we extracted a small subset from the MNIST dataset consisting of 7's and 9's.

<img src="https://github.com/PeterHolderrieth/backpropagation/blob/master/plots/illustrate_7_9_mnist.png">




